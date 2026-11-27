# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Kiwoong Park

"""Isaac collection loop around original SuperMimic WM and imagined BPTT learners."""

from dataclasses import dataclass, field
import time
import torch
from tensordict import TensorDict
from .. import upstream  # noqa: F401  # Expose the pinned SuperMimic source before upstream imports.
from supermimic.models import Dynamics, MultiEncoderAE, build_mlp
from supermimic.representations import create_representation
from supermimic.replay_buffer import ReplayBuffer, FieldSpec
from supermimic.tasks.task_spec import make_motion_task, build_actor_obs_td, resolve_task_specs, TasksCfg
from supermimic.envs.world_model_env import WorldModelEnv, sample_start
from supermimic.learners.wm_learner import WorldModelLearner, WorldModelLearnerCfg
from supermimic.learners.policy_learner import (
    ActorCriticPolicyLearner,
    ActorCriticPolicyLearnerCfg,
)
from cyclo_lab.utils.episode_metrics import EpisodeMetrics
from .. import export
from ..reference.bridge import IsaacBridge, BODIES
from ..reference.ik import build_urdf_analytic_ik


@dataclass
class RunnerCfg:
    log_interval: int = 10
    representation: str = "gencoord"
    actor_hidden_dims: tuple[int, ...] = (256, 256)
    dynamics_hidden_dims: tuple[int, ...] = (256, 256)
    critic_hidden_dims: tuple[int, ...] = (256, 256)
    latent_dim: int = 32
    wm: WorldModelLearnerCfg = field(
        default_factory=lambda: WorldModelLearnerCfg(num_batches=1, batch_size=64, horizon=8)
    )
    policy: ActorCriticPolicyLearnerCfg = field(
        default_factory=lambda: ActorCriticPolicyLearnerCfg(num_batches=1, batch_size=64, horizon=16)
    )
    replay_ratio: int = 16
    buffer_size: int = 1024
    eager: bool = False
    history_length: int = 1
    actor_obs: str = "rep_state"
    slot_mix: dict[str, float] = field(default_factory=lambda: {"motion": 1.0, "velocity": 0.0})
    latent_bc_weight: float = 0.1
    termination_softness: float = 0.05


@dataclass
class GenCoordCfg(RunnerCfg):
    wm: WorldModelLearnerCfg = field(
        default_factory=lambda: WorldModelLearnerCfg(num_batches=10, batch_size=512, horizon=8)
    )
    policy: ActorCriticPolicyLearnerCfg = field(
        default_factory=lambda: ActorCriticPolicyLearnerCfg(num_batches=1, batch_size=256, horizon=32)
    )
    actor_hidden_dims: tuple = (2048, 1024, 512, 256)
    dynamics_hidden_dims: tuple = (2048, 2048, 2048)
    critic_hidden_dims: tuple = (256, 128)
    actor_obs: str = "proprio"
    history_length: int = 10


@dataclass
class MaxCoordCfg(GenCoordCfg):
    representation: str = "maxcoord"


class IsaacSuperMimicRunner:
    actor_export = export
    family = "supermimic"

    def __init__(self, env, cfg: RunnerCfg):
        if min(cfg.wm.batch_size, cfg.wm.num_batches, cfg.policy.batch_size, cfg.policy.num_batches) < 1:
            raise ValueError("Learner batch sizes and counts must be positive")
        if cfg.wm.horizon < 2 or cfg.policy.batch_size * cfg.policy.horizon < 4:
            raise ValueError("WM needs >=2 frames; actor-critic needs >=4 rollout samples")
        if cfg.buffer_size < cfg.wm.horizon:
            raise ValueError("Replay buffer smaller than WM horizon")
        if cfg.representation not in ("gencoord", "maxcoord"):
            raise ValueError("Unknown SuperMimic representation")
        self.cfg, self.env = cfg, env
        self.bridge = IsaacBridge(env, cfg.representation)
        b = self.bridge
        self.device = b.device
        self.metrics = EpisodeMetrics(env.unwrapped.num_envs, self.device)
        self.iteration = 0
        self.collected_steps = 0
        self.rep = create_representation(
            cfg.representation,
            b.full_state_dim,
            23,
            b.step_dt,
            env.unwrapped.cfg.decimation,
            b.robot.data.default_joint_pos.torch[0].clone(),
            cfg.actor_obs,
        ).to(self.device)
        limits = b.robot.data.soft_joint_pos_limits.torch[0].clone()
        context_width = b.context().shape[-1]
        if cfg.representation == "maxcoord":
            self.rep.bind_native(
                limits,
                build_urdf_analytic_ik(b.env.cfg.scene.robot.spawn.asset_path, b.robot.joint_names, BODIES, self.device),
                (slice(4, None), b.sim_context) if context_width > 1 else None,
            )
        else:
            self.rep.bind_native(limits, b.fk)
        self.scale, self.offset = b.action_transform()
        self.dynamics = Dynamics(
            self.rep, 23, context_width, self.scale, self.offset, cfg.dynamics_hidden_dims
        ).to(self.device)
        goal, _ = self.rep.compute_wm_goal(b.motion, b.frame_ids(), self.rep.extract_aux_state(b.state()))
        self.specs = [make_motion_task(self.rep, goal.shape[-1])]
        self.play_velocity = None
        self.training_noise = True
        self.history = None
        self.reset_rows = None
        self.slot_mix = {"motion": 1.0}
        sensor_width = self.rep.rep_state_dim
        if cfg.actor_obs == "proprio":
            sensor_width = self.rep.proprio_obs_dim(23)
            self.specs = resolve_task_specs(TasksCfg(), self.rep, goal.shape[-1], self.device)
            self.slot_mix = cfg.slot_mix.copy()
        elif cfg.slot_mix.get("velocity", 0) != 0:
            raise ValueError("The velocity encoder requires actor_obs='proprio'")
        self.actor = MultiEncoderAE(
            sensor_width,
            23,
            self.specs,
            self.slot_mix,
            cfg.latent_dim,
            cfg.actor_hidden_dims,
            cfg.latent_bc_weight,
            cfg.history_length,
        ).to(self.device)
        for spec, (start, end) in zip(self.specs, self.actor.slot_ranges(cfg.policy.batch_size), strict=True):
            if self.slot_mix[spec.name] > 0 and start == end:
                raise ValueError(
                    f"No training samples for {spec.name}; adjust batch size or encoder fraction"
                )
        self.critic = build_mlp(
            self.rep.rep_state_dim + goal.shape[-1] + context_width, cfg.critic_hidden_dims, 1
        ).to(self.device)
        self.buffer = ReplayBuffer(
            {
                **({"actor_obs": FieldSpec(sensor_width)} if cfg.actor_obs == "proprio" else {}),
                "state": FieldSpec(b.full_state_dim),
                "action": FieldSpec(23),
                "prev_action": FieldSpec(23),
                "sim_ctx": FieldSpec(context_width),
                "motion_time": FieldSpec(dtype=torch.long),
                "terminated": FieldSpec(dtype=torch.bool),
                "truncated": FieldSpec(dtype=torch.bool),
            },
            cfg.buffer_size,
            env.unwrapped.num_envs,
            self.device,
        )
        self.wm_env = WorldModelEnv(
            self.dynamics,
            self.rep,
            b.world_model_facade(),
            self.specs,
            history_length=cfg.history_length,
            termination_softness=cfg.termination_softness,
        )
        self.sampler = None
        if context_width > 1:
            from .reset_sampling import ReturnResetSampler

            command = b.env.command_manager.get_term("reference_trajectory")
            self.sampler = command.sampler = ReturnResetSampler(b.env, command, self.wm_env.max_step_reward)
        self.wm_learner = WorldModelLearner(
            self.dynamics,
            self.rep,
            cfg.wm,
            self.device,
        )
        self.policy_learner = ActorCriticPolicyLearner(
            self.actor,
            self.critic,
            self.wm_env,
            cfg.policy,
            self.device,
        )
        if cfg.eager:
            # Same loss/gradient/optimizer steps, without compilation/CUDA graph capture.
            self.wm_learner._compiled_rollout = self.wm_learner._rollout
            self.wm_learner._compiled_compute_losses = self.wm_learner._compute_losses
            self.wm_learner._compiled_represent = self.rep.represent
            self.wm_learner._captured_update = self.wm_learner._update_from
            self.policy_learner._rollout_actor = self.actor
            self.policy_learner._captured_update = self.policy_learner._update_from

    def reference_frames(self):
        frames = self.bridge.frame_ids()
        if self.cfg.actor_obs == "rep_state" or hasattr(self.bridge.env, "encoder_bias"):
            return frames
        ends = self.bridge.motion.motion_ends
        # MJLab advances the command once after resetting at the physical reference frame.
        return (frames + 1).minimum(ends[torch.bucketize(frames, ends, right=True)] - 1)

    def actor_obs(self, state, frames):
        aux = self.rep.extract_aux_state(state)
        rep = self.rep.represent(state)
        goals = {
            spec.name: spec.extract_training_goal(self.bridge.motion, frames, aux) for spec in self.specs
        }
        if self.play_velocity is not None:
            goals["velocity"] = self.play_velocity.expand(len(state), -1)
        if self.cfg.actor_obs == "rep_state":
            return build_actor_obs_td(goals, rep[:, None])
        robot = self.bridge.robot.data
        position = robot.joint_pos.torch
        if self.cfg.representation == "gencoord" and hasattr(self.bridge.env, "encoder_bias"):
            position = position + self.bridge.env.encoder_bias
        joints = torch.cat((position, robot.joint_vel.torch), -1)
        previous = self.env.unwrapped.action_manager.action
        if self.reset_rows is not None:
            previous = previous.masked_fill(self.reset_rows[:, None], 0)
        self.clean_sensor = self.rep.compute_actor_sensor(rep, self.bridge.context(), aux, joints, previous)
        sensor = self.rep.add_proprio_noise(self.clean_sensor) if self.training_noise else self.clean_sensor
        seed = sensor[:, None].expand(-1, self.cfg.history_length, -1)
        if self.history is None:
            self.history = seed
        else:
            self.history = torch.where(
                self.reset_rows[:, None, None], seed, torch.cat((self.history[:, 1:], sensor[:, None]), 1)
            )
        return build_actor_obs_td(goals, self.history)

    def start_playback(self, encoder, velocity_command, motions):
        self.actor.set_slot_mix({encoder: 1.0})
        if encoder == "velocity":
            self.play_velocity = torch.tensor(velocity_command or (0.0, 0.0, 0.0), device=self.device)
        from ..viewer import create_play_viewer

        return create_play_viewer(self, motions)

    @torch.no_grad()
    def play_step(self):
        self.actor.eval()
        resets = self.bridge.reset_counts()
        obs = self.actor_obs(self.bridge.state(), self.reference_frames())
        _, _, term, trunc, _ = self.env.step(self.actor(obs))
        self.reset_rows = term | trunc | (self.bridge.reset_counts() != resets)

    def collect(self, steps):
        collected = []
        for _ in range(steps):
            # Snapshot BEFORE env.step: that call can reset and advance motion in place.
            state, frames = self.bridge.state(), self.reference_frames()
            resets = self.bridge.reset_counts()
            with torch.no_grad():
                self.actor.eval()
                obs = self.actor_obs(state, frames)
                previous_action = self.env.unwrapped.action_manager.action.clone()
                action = self.actor(obs)
                context = self.bridge.context()
                if not torch.isfinite(action).all() or not torch.isfinite(state).all():
                    raise RuntimeError("Nonfinite real state/action before collection")
                _, reward, terminated, truncated, extras = self.env.step(action)
                # Commands can teleport at clip end without an env done. Never
                # teach the WM that a resample is a physical transition.
                resampled = self.bridge.reset_counts() != resets
                if self.sampler is not None:
                    ends = self.bridge.motion.motion_ends
                    clip_end = torch.isin(frames + 1, ends) | (resampled & ~terminated & ~truncated)
                    self.sampler.record(
                        frames, reward, terminated & ~truncated, clip_end, truncated & ~clip_end
                    )
                truncated = truncated | (resampled & ~terminated)
                self.reset_rows = terminated | truncated
                self.metrics.record(reward, self.reset_rows, extras=extras)
                transition = TensorDict(
                    dict(
                        **({"actor_obs": self.clean_sensor} if self.cfg.actor_obs == "proprio" else {}),
                        state=state,
                        action=action.clone(),
                        prev_action=previous_action,
                        motion_time=frames,
                        sim_ctx=context,
                        terminated=terminated.clone(),
                        truncated=truncated.clone(),
                    ),
                    batch_size=[len(state)],
                    device=self.device,
                )
                self.buffer.insert(transition)
                collected.append((state, action, obs))
                self.collected_steps += 1
        self.wm_learner.update_normalization(
            torch.cat([x[0] for x in collected]), torch.cat([x[1] for x in collected])
        )
        self.actor.train()
        for _, _, obs in collected:
            self.actor.update_normalization(obs)

    @property
    def active_encoders(self):
        return tuple(name for name, fraction in self.slot_mix.items() if fraction > 0)

    def actor_checkpoint(self):
        return {"actor": self.actor.state_dict()}

    def restore_actor(self, bundle, *, training=False):
        self.actor.load_state_dict(bundle["actor"])

    def checkpoint_optimizers(self):
        return dict(
            dynamics=self.wm_learner.optimizers,
            actor=self.policy_learner.actor_optimizers,
            critic=self.policy_learner.critic_optimizers,
        )

    def reset_playback_history(self):
        self.history = None
        self.reset_rows = None

    def train_iteration(self, collection_steps=None):
        collection_start = time.perf_counter()
        steps = 1 if collection_steps is None else collection_steps
        self.collect(steps)
        episode_logs = self.metrics.scalars()
        # Resolve collected GPU work before attributing time to learning.
        if torch.device(self.device).type == "cuda":
            torch.cuda.synchronize(self.device)
        collection_time = time.perf_counter() - collection_start
        ratio = (
            self.bridge.env.num_envs
            * self.cfg.replay_ratio
            / (self.cfg.wm.num_batches * self.cfg.wm.batch_size * self.cfg.wm.horizon)
        )
        updates = int((self.iteration + 1) * ratio) - int(self.iteration * ratio)
        if collection_steps is not None:
            updates = 1
        updates_logs = []
        if self.buffer.num_transitions >= self.cfg.wm.horizon:
            for _ in range(updates):
                updates_logs.append(self._update_models())
        logs = {}
        if updates_logs:
            for key in updates_logs[0]:
                total = sum(values[key] for values in updates_logs)
                logs[key] = total if key.startswith("Perf/") else total / len(updates_logs)
        else:
            logs.update({
                "Perf/world_model_learning_time": 0.0,
                "Perf/policy_learning_time": 0.0,
                "Perf/learning_time": 0.0,
            })
        logs["Perf/collection_time"] = collection_time
        self.iteration += 1
        return logs | episode_logs

    def update(self):
        logs = self._update_models()
        self.iteration += 1
        return logs

    def _update_models(self):
        if self.buffer.num_transitions < self.cfg.wm.horizon:
            raise RuntimeError("Collect enough Isaac transitions before learning")
        wm_start = time.perf_counter()
        wm = self.wm_learner.update(
            self.buffer.sample(self.cfg.wm.horizon, self.cfg.wm.batch_size)
            for _ in range(self.cfg.wm.num_batches)
        )
        wm = {key: float(value) for key, value in wm.items()}
        wm_time = time.perf_counter() - wm_start
        policy_cfg = self.policy_learner.cfg
        policy_start = time.perf_counter()
        policy = self.policy_learner.update(
            sample_start(self.buffer, self.cfg.history_length, policy_cfg.batch_size)
            for _ in range(policy_cfg.num_batches)
        )
        policy = {key: float(value) for key, value in policy.items()}
        policy_time = time.perf_counter() - policy_start
        return wm | policy | {
            "Perf/world_model_learning_time": wm_time,
            "Perf/policy_learning_time": policy_time,
            "Perf/learning_time": wm_time + policy_time,
        }
