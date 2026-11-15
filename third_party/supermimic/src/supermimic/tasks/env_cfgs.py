"""Environment configurations for SuperMimic representations."""

from __future__ import annotations

from typing import cast

import mjlab.envs.mdp as mdp
import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from supermimic.representations import (
  PROPRIO_NOISE_HALF_WIDTHS,
  assemble_anchor_window,
  motion_anchor_window,
  motion_goal_window,
)

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _make_asset_obs_fn(attr_name: str, *, subtract_origins: bool = False):
  """Factory for observation terms that read a single asset data attribute."""

  def fn(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
  ) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    val = getattr(asset.data, attr_name)
    return val - env.scene.env_origins if subtract_origins else val

  return fn


root_pos = _make_asset_obs_fn("root_link_pos_w", subtract_origins=True)
root_ori = _make_asset_obs_fn("root_link_quat_w")
root_lin_vel = _make_asset_obs_fn("root_link_lin_vel_w")
root_ang_vel = _make_asset_obs_fn("root_link_ang_vel_w")
joint_pos = _make_asset_obs_fn("joint_pos")
joint_vel = _make_asset_obs_fn("joint_vel")


# ---------------------------------------------------------------------------
# Maximal coordinate observation terms
# ---------------------------------------------------------------------------


def _make_body_obs_fn(attr_name: str, *, subtract_origins: bool = False):
  """Factory for observation terms that read a MotionCommand body attribute."""

  def fn(env: ManagerBasedRlEnv, command_name: str = "motion") -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    val = getattr(command, attr_name)
    if subtract_origins:
      val = val - env.scene.env_origins.unsqueeze(1)
    return val.reshape(env.num_envs, -1)

  return fn


body_pos_w = _make_body_obs_fn("robot_body_pos_w", subtract_origins=True)
body_quat_w = _make_body_obs_fn("robot_body_quat_w")
body_lin_vel_w = _make_body_obs_fn("robot_body_lin_vel_w")
body_ang_vel_w = _make_body_obs_fn("robot_body_ang_vel_w")


def motion_command_window(
  env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
  """Goal command: joint targets over a future window of reference frames."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return motion_goal_window(command.motion, command.time_steps)


def _motion_anchor_rel(
  env: ManagerBasedRlEnv, command_name: str
) -> tuple[torch.Tensor, torch.Tensor]:
  """Windowed motion anchor as ``(rel_pos, rel_ori)`` in the robot root frame.

  ``env_origins`` is added so the motion anchor shares ``robot_anchor_pos_w``'s
  world frame before the relative transform.
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  target_pos, target_quat = motion_anchor_window(command.motion, command.time_steps)
  target_pos = target_pos + env.scene.env_origins[:, None, :]
  return assemble_anchor_window(
    command.robot_anchor_pos_w, command.robot_anchor_quat_w, target_pos, target_quat
  )


def motion_anchor_pos_b(
  env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
  """Windowed anchor translation relative to the robot root (world-model only)."""
  return _motion_anchor_rel(env, command_name)[0]


def motion_anchor_ori_b(
  env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
  """Windowed anchor orientation (rot6d) relative to the robot root."""
  return _motion_anchor_rel(env, command_name)[1]


# ---------------------------------------------------------------------------
# Configuration factories
# ---------------------------------------------------------------------------


def _add_term(terms: dict[str, ObservationTermCfg], name: str, func, **params) -> None:
  """Insert a proprio term keyed by ``name``; noise pulled from the shared spec."""
  half = PROPRIO_NOISE_HALF_WIDTHS[name]
  kwargs: dict = {"func": func}
  if params:
    kwargs["params"] = params
  if half > 0:
    kwargs["noise"] = Unoise(n_min=-half, n_max=half)
  terms[name] = ObservationTermCfg(**kwargs)


def _sensor_actor_group() -> ObservationGroupCfg:
  """Sensor-only proprio for the actor — what the deployed robot reads.

  Layout: ``[projected_gravity(3), base_ang_vel(3), joint_pos_rel(N),
  joint_vel(N), actions(A)]``. Motion targets live in the ``goal`` group.
  ``enable_corruption=False``: noise is owned by
  ``representations.proprio_noise_bounds`` so train and BPTT rollout
  share one noise spec.
  """
  terms: dict[str, ObservationTermCfg] = {}
  _add_term(terms, "projected_gravity", mdp.projected_gravity)
  _add_term(terms, "base_ang_vel", mdp.builtin_sensor, sensor_name="robot/imu_ang_vel")
  _add_term(terms, "joint_pos", mdp.joint_pos_rel, biased=True)
  _add_term(terms, "joint_vel", mdp.joint_vel_rel)
  _add_term(terms, "actions", mdp.last_action)
  return ObservationGroupCfg(
    terms=terms, concatenate_terms=True, enable_corruption=False
  )


def _goal_obs_group() -> ObservationGroupCfg:
  """Goal group, layout ``[command, rel_pos, rel_ori]`` matching wm_goal.

  ``motion_anchor_pos_b`` (rel_pos) is world-model only; the deployed actor drops
  it by name (see the exporter), so pos and ori stay separate terms.
  """
  return ObservationGroupCfg(
    terms={
      "command": ObservationTermCfg(func=motion_command_window),
      "motion_anchor_pos_b": ObservationTermCfg(func=motion_anchor_pos_b),
      "motion_anchor_ori_b": ObservationTermCfg(func=motion_anchor_ori_b),
    },
    concatenate_terms=True,
    enable_corruption=False,
  )


def gencoord_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create generalized-coordinate world model environment configuration based on G1 tracking."""
  cfg = unitree_g1_flat_tracking_env_cfg(play=play)

  cfg.commands["motion"].anchor_body_name = "pelvis"

  cfg.observations["actor"] = _sensor_actor_group()

  # Add full state observation group
  cfg.observations["full_state"] = ObservationGroupCfg(
    terms={
      "root_pos": ObservationTermCfg(func=root_pos),
      "root_ori": ObservationTermCfg(func=root_ori),
      "joint_pos": ObservationTermCfg(func=joint_pos),
      "root_lin_vel": ObservationTermCfg(func=root_lin_vel),
      "root_ang_vel": ObservationTermCfg(func=root_ang_vel),
      "joint_vel": ObservationTermCfg(func=joint_vel),
    },
    concatenate_terms=True,
    enable_corruption=False,
  )

  # Add goal observation group
  cfg.observations["goal"] = _goal_obs_group()

  cfg.events = {
    name: term
    for name, term in cfg.events.items()
    if name in {"foot_friction", "base_com", "encoder_bias"}
  }

  cfg.rewards.pop("self_collisions", None)
  cfg.rewards["joint_limit"].weight = -2.0
  # cfg.rewards.pop("motion_global_root_pos", None)
  # cfg.rewards.pop("motion_global_root_ori", None)

  cfg.terminations.pop("ee_body_pos", None)

  return cfg


def maxcoord_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create maximal-coordinate world model environment configuration.

  Extends the base gencoord_env_cfg by replacing the full_state and goal
  observation groups with per-body maximal coordinate versions.
  """
  cfg = gencoord_env_cfg(play=play)

  # Override full state: per-body maximal coordinates
  # Layout: [body_pos(B*3), body_quat(B*4), body_lin_vel(B*3), body_ang_vel(B*3)]
  cfg.observations["full_state"] = ObservationGroupCfg(
    terms={
      "body_pos_w": ObservationTermCfg(
        func=body_pos_w,
        params={"command_name": "motion"},
      ),
      "body_quat_w": ObservationTermCfg(
        func=body_quat_w,
        params={"command_name": "motion"},
      ),
      "body_lin_vel_w": ObservationTermCfg(
        func=body_lin_vel_w,
        params={"command_name": "motion"},
      ),
      "body_ang_vel_w": ObservationTermCfg(
        func=body_ang_vel_w,
        params={"command_name": "motion"},
      ),
    },
    concatenate_terms=True,
    enable_corruption=False,
  )

  cfg.observations["goal"] = _goal_obs_group()

  return cfg
