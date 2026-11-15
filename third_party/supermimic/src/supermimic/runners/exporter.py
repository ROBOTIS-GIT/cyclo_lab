"""ONNX exporter for the multi-encoder actor.

One ONNX per task encoder, each carrying the deploy contract that task
needs at inference:

- **motion**: mjlab tracking contract — bundles the motion clip arrays as
  buffers so the deploy can recover `(joint_pos, body_*_w)` per ``time_step``.
  Graph: ``(obs, time_step) → (action, joint_pos, joint_vel, body_pos_w,
  body_quat_w, body_lin_vel_w, body_ang_vel_w)``.
- **velocity**: joystick contract — no motion baked in; the user-supplied
  command is part of ``obs``. Graph: ``(obs) → action`` where
  ``obs = cat([history, command])`` and ``command_names = ["twist"]``.
- **velocity_replay**: sim-to-real sanity-check variant of velocity. Same
  output contract as motion (action + motion buffers), but the velocity
  command is precomputed offline from the motion clip via the velocity
  task's own ``precompute_replay_goal`` (noise-free, same clamping as training).
"""

from __future__ import annotations

import copy
import os
from typing import cast

import torch
import torch.nn as nn
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.tasks.tracking.mdp import MotionCommand

from supermimic.models import MultiEncoderAE
from supermimic.representations import term_major_to_temporal


def _export_onnx(
  module: nn.Module,
  sample_inputs: tuple[torch.Tensor, ...],
  input_names: list[str],
  output_names: list[str],
  path: str,
) -> None:
  # Export a CPU copy; captured training updates read the live parameters by address.
  module = copy.deepcopy(module).cpu().eval()
  torch.onnx.export(
    module,
    sample_inputs,
    path,
    export_params=True,
    opset_version=18,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes={},
    dynamo=False,
  )


def _export_module(
  module: nn.Module,
  sample_inputs: tuple[torch.Tensor, ...],
  input_names: list[str],
  output_names: list[str],
  path: str,
  filename: str,
) -> str:
  """makedirs + ONNX export; returns the written path."""
  os.makedirs(path, exist_ok=True)
  onnx_path = os.path.join(path, filename)
  _export_onnx(module, sample_inputs, input_names, output_names, onnx_path)
  return onnx_path


class _SingleEncoderOnnx(nn.Module):
  """Single-task slice of the multi-encoder actor: ``cat([history, goal]) → action``.

  Bakes the chosen encoder + shared decoder + the matching normalizers into
  one ONNX-compatible module. The deploy-time latent (``tanh(enc_out)``) is
  applied here rather than read from the actor's forward
  to keep the graph closed and free of training-only branches.

  The obs part of the input is the TERM-MAJOR ``history_length``-frame stack —
  mjlab's native layout, ``[term0_t0..tH-1, term1_t0..tH-1, ...]`` — converted
  internally to the frame-major window the decoder consumes; the encoder
  consumes only the goal.
  """

  def __init__(
    self, actor: MultiEncoderAE, task_name: str, term_widths: list[int]
  ) -> None:
    super().__init__()
    self.obs_normalizer = actor.obs_normalizer
    self.goal_normalizer = actor.goal_normalizers[task_name]
    self.encoder = actor.encoders[task_name]
    self.decoder = actor.decoder
    self.history_length = actor.history_length
    self.hist_size = actor.history_length * actor.obs_dim
    self.term_widths = term_widths
    self.goal_dim = next(ts for ts in actor.task_specs if ts.name == task_name).goal_dim
    self.input_size = self.hist_size + self.goal_dim

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    history = term_major_to_temporal(
      x[..., : self.hist_size], self.term_widths, self.history_length
    )
    goal = x[..., self.hist_size :]
    history_n = self.obs_normalizer(history)
    z = torch.tanh(self.encoder(self.goal_normalizer(goal)))
    return self.decoder(torch.cat([z, history_n.flatten(start_dim=1)], dim=-1))


_MOTION_OUTPUT_NAMES = [
  "actions",
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
]


class _MotionDeployWrapper(nn.Module):
  """Encoder slice + motion-clip lookup tables.

  Output tuple matches the mjlab tracking deploy contract: ``(obs, time_step)
  → (action, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w,
  body_ang_vel_w)``.

  With ``command_table`` set, the per-frame command baked from the motion clip
  is looked up at ``time_step`` and concatenated onto the sensor obs before the
  encoder (the velocity-replay sim-to-real sanity check). With ``None`` the
  encoder consumes ``obs`` directly — the motion task, whose goal is already
  part of ``obs``.
  """

  def __init__(
    self,
    actor: MultiEncoderAE,
    env: ManagerBasedRlEnv,
    task_name: str,
    term_widths: list[int],
    command_table: torch.Tensor | None = None,
  ) -> None:
    super().__init__()
    self.actor = _SingleEncoderOnnx(actor, task_name, term_widths)
    self._has_command = command_table is not None
    # ONNX ``obs`` width: history-only when a command is appended below, else
    # the full [history, goal] the encoder slice expects.
    self.obs_input_size = (
      self.actor.hist_size if self._has_command else self.actor.input_size
    )
    if self._has_command:
      self.register_buffer("_command_table", command_table.detach().cpu())

    motion = cast(MotionCommand, env.command_manager.get_term("motion")).motion
    frames = torch.arange(motion.time_step_total, device=motion.joint_pos.device)
    body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w = motion.body_state(frames)
    self.register_buffer("_joint_pos", motion.joint_pos.detach().cpu())
    self.register_buffer("_joint_vel", motion.joint_vel.detach().cpu())
    self.register_buffer("_body_pos_w", body_pos_w.detach().cpu())
    self.register_buffer("_body_quat_w", body_quat_w.detach().cpu())
    self.register_buffer("_body_lin_vel_w", body_lin_vel_w.detach().cpu())
    self.register_buffer("_body_ang_vel_w", body_ang_vel_w.detach().cpu())
    self._time_step_total = int(self._joint_pos.shape[0])

  def forward(
    self, x: torch.Tensor, time_step: torch.Tensor
  ) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
  ]:
    t = torch.clamp(time_step.long().squeeze(-1), max=self._time_step_total - 1)
    actor_in = (
      torch.cat([x, self._command_table[t]], dim=-1) if self._has_command else x
    )
    return (
      self.actor(actor_in),
      self._joint_pos[t],
      self._joint_vel[t],
      self._body_pos_w[t],
      self._body_quat_w[t],
      self._body_lin_vel_w[t],
      self._body_ang_vel_w[t],
    )


def _tracking_metadata(env: ManagerBasedRlEnv, run_name: str) -> dict:
  """Base metadata + the mjlab tracking deploy fields (anchor + body names)."""
  motion_term = cast(MotionCommand, env.command_manager.get_term("motion"))
  metadata = get_base_metadata(env, run_name)
  metadata["anchor_body_name"] = motion_term.cfg.anchor_body_name
  metadata["body_names"] = list(motion_term.cfg.body_names)
  return metadata


def _export_motion_graph(
  wrapper: _MotionDeployWrapper,
  path: str,
  filename: str,
) -> str:
  return _export_module(
    wrapper,
    (torch.zeros(1, wrapper.obs_input_size), torch.zeros(1, 1)),
    ["obs", "time_step"],
    _MOTION_OUTPUT_NAMES,
    path,
    filename,
  )


def _export_motion(
  actor: MultiEncoderAE,
  env: ManagerBasedRlEnv,
  path: str,
  run_name: str,
  filename: str,
  term_widths: list[int],
) -> None:
  wrapper = _MotionDeployWrapper(actor, env, "motion", term_widths)
  onnx_path = _export_motion_graph(wrapper, path, filename)
  metadata = _tracking_metadata(env, run_name)
  actor_terms = list(metadata["observation_names"])
  actor_goal_names = [
    n
    for n in env.observation_manager.active_terms["goal"]
    if n != "motion_anchor_pos_b"
  ]
  metadata["observation_names"] = actor_terms + actor_goal_names
  metadata["observation_history_lengths"] = [actor.history_length] * len(
    actor_terms
  ) + [1] * len(actor_goal_names)
  attach_metadata_to_onnx(onnx_path, metadata)


def _export_velocity_replay(
  actor: MultiEncoderAE,
  env: ManagerBasedRlEnv,
  path: str,
  run_name: str,
  filename: str,
  term_widths: list[int],
) -> str | None:
  velocity_task = next((ts for ts in actor.task_specs if ts.name == "velocity"), None)
  if velocity_task is None or velocity_task.precompute_replay_goal is None:
    return None
  motion = cast(MotionCommand, env.command_manager.get_term("motion")).motion
  command_table = velocity_task.precompute_replay_goal(motion)
  wrapper = _MotionDeployWrapper(actor, env, "velocity", term_widths, command_table)
  onnx_path = _export_motion_graph(wrapper, path, filename)
  metadata = _tracking_metadata(env, run_name)
  metadata["observation_history_lengths"] = [actor.history_length] * len(
    metadata["observation_names"]
  )
  attach_metadata_to_onnx(onnx_path, metadata)
  return onnx_path


def _export_velocity(
  actor: MultiEncoderAE,
  env: ManagerBasedRlEnv,
  path: str,
  run_name: str,
  filename: str,
  term_widths: list[int],
) -> None:
  wrapper = _SingleEncoderOnnx(actor, "velocity", term_widths)
  onnx_path = _export_module(
    wrapper,
    (torch.zeros(1, wrapper.input_size),),
    ["obs"],
    ["actions"],
    path,
    filename,
  )
  metadata = get_base_metadata(env, run_name)
  actor_terms = list(metadata["observation_names"])
  metadata["observation_names"] = actor_terms + ["command"]
  metadata["command_names"] = ["twist"]
  metadata["observation_history_lengths"] = [actor.history_length] * len(
    actor_terms
  ) + [1]
  attach_metadata_to_onnx(onnx_path, metadata)


_TASK_EXPORTERS = {
  "motion": _export_motion,
  "velocity": _export_velocity,
}


def export_actor_as_onnx(
  actor: MultiEncoderAE,
  env: ManagerBasedRlEnv,
  path: str,
  run_name: str,
  term_widths: list[int],
) -> list[str]:
  """Export one ``policy_<task>.onnx`` per encoder in ``actor``.

  Returns the list of file paths produced. Unknown task names are skipped
  with a warning rather than erroring.
  """
  written: list[str] = []
  for ts in actor.task_specs:
    fn = _TASK_EXPORTERS.get(ts.name)
    if fn is None:
      print(f"[WARN] No ONNX exporter registered for task '{ts.name}'; skipping.")
      continue
    filename = f"policy_{ts.name}.onnx"
    fn(actor, env, path, run_name, filename, term_widths)
    written.append(os.path.join(path, filename))
  replay = _export_velocity_replay(
    actor, env, path, run_name, "policy_velocity_replay.onnx", term_widths
  )
  if replay is not None:
    written.append(replay)
  return written
