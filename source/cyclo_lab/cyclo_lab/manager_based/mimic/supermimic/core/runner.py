# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Isaac collection loop around original SuperMimic WM and imagined BPTT learners."""

from dataclasses import dataclass
import torch
from tensordict import TensorDict
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
from . import export
from .bridge import IsaacBridge, BODIES, URDF
from .ik import build_urdf_analytic_ik


@dataclass
class RunnerCfg:
    representation: str = "gencoord"
    hidden: tuple[int, ...] = (256, 256)
    world_hidden: tuple[int, ...] = ()
    critic_hidden: tuple[int, ...] = ()
    latent: int = 32
    batch_size: int = 64
    wm_batches: int = 1
    wm_batch_size: int | None = None
    replay_ratio: int = 16
    horizon: int = 8
    policy_horizon: int = 16
    buffer_size: int = 1024
    collection_steps: int = 16
    eager: bool = False
    history_length: int = 1
    actor_obs: str = "rep_state"
    velocity_fraction: float = 0.0


@dataclass
class GenCoordCfg(RunnerCfg):
    batch_size: int = 256
    wm_batches: int = 10
    wm_batch_size: int | None = 512
    policy_horizon: int = 32
    hidden: tuple = (2048, 1024, 512, 256)
    world_hidden: tuple = (2048, 2048, 2048)
    critic_hidden: tuple = (256, 128)
    actor_obs: str = "proprio"
    history_length: int = 10


@dataclass
class MaxCoordCfg(GenCoordCfg):
    representation: str = "maxcoord"


class EncoderBias:
    @staticmethod
    def denormalize(context, columns):
        return context[..., columns] * 0.01


class IsaacSuperMimicRunner:
    actor_export = export
    family = "supermimic"

    def __init__(self, env, cfg: RunnerCfg):
        if cfg.collection_steps < 1 or cfg.batch_size < 1:
            raise ValueError("Collection steps and batch size must be positive")
        if cfg.horizon < 2 or cfg.batch_size * cfg.policy_horizon < 4:
            raise ValueError("WM needs >=2 frames; actor-critic needs >=4 rollout samples")
        if cfg.buffer_size < cfg.horizon:
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
                build_urdf_analytic_ik(URDF, b.robot.joint_names, BODIES, self.device),
                (slice(4, None), EncoderBias()) if context_width > 1 else None,
            )
        else:
            self.rep.bind_native(limits, b.fk)
        self.scale, self.offset = b.action_transform()
        self.dynamics = Dynamics(
            self.rep, 23, context_width, self.scale, self.offset, cfg.world_hidden or cfg.hidden
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
            self.slot_mix = {"motion": 1.0 - cfg.velocity_fraction, "velocity": cfg.velocity_fraction}
        self.actor = MultiEncoderAE(
            sensor_width,
            23,
            self.specs,
            self.slot_mix,
            cfg.latent,
            cfg.hidden,
            0.1 if cfg.velocity_fraction else 0.0,
            cfg.history_length,
        ).to(self.device)
        for spec, (start, end) in zip(self.specs, self.actor.slot_ranges(cfg.batch_size), strict=True):
            if self.slot_mix[spec.name] > 0 and start == end:
                raise ValueError(
                    f"No training samples for {spec.name}; adjust batch size or encoder fraction"
                )
        self.critic = build_mlp(
            self.rep.rep_state_dim + goal.shape[-1] + context_width, cfg.critic_hidden or cfg.hidden, 1
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
            termination_softness=0.05,
        )
        self.sampler = None
        if context_width > 1:
            from .reset_sampling import ReturnResetSampler

            command = b.env.command_manager.get_term("reference_trajectory")
            self.sampler = command.sampler = ReturnResetSampler(b.env, command, self.wm_env.max_step_reward)
        self.wm_learner = WorldModelLearner(
            self.dynamics,
            self.rep,
            WorldModelLearnerCfg(
                num_batches=cfg.wm_batches,
                batch_size=cfg.wm_batch_size or cfg.batch_size,
                horizon=cfg.horizon,
            ),
            self.device,
        )
        self.policy_learner = ActorCriticPolicyLearner(
            self.actor,
            self.critic,
            self.wm_env,
            ActorCriticPolicyLearnerCfg(num_batches=1, batch_size=cfg.batch_size, horizon=cfg.policy_horizon),
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
        if self.bridge.env.sim.has_gui:
            from .viewer import create_play_viewer

            return create_play_viewer(self, motions)

    @torch.no_grad()
    def play_step(self):
        self.training_noise = False
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
                _, reward, terminated, truncated, _ = self.env.step(action)
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
                self.metrics.record(reward, self.reset_rows)
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

    def restore_actor(self, bundle):
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
        steps = 1 if collection_steps is None else collection_steps
        self.collect(steps)
        ratio = (
            self.bridge.env.num_envs
            * self.cfg.replay_ratio
            / (self.cfg.wm_batches * (self.cfg.wm_batch_size or self.cfg.batch_size) * self.cfg.horizon)
        )
        updates = int((self.iteration + 1) * ratio) - int(self.iteration * ratio)
        if collection_steps is not None:
            updates = 1
        logs = {}
        if self.buffer.num_transitions >= self.cfg.horizon:
            for _ in range(updates):
                logs = self._update_models()
        self.iteration += 1
        return logs | self.metrics.scalars()

    def update(self):
        logs = self._update_models()
        self.iteration += 1
        return logs

    def _update_models(self):
        if self.buffer.num_transitions < self.cfg.horizon:
            raise RuntimeError("Collect enough Isaac transitions before learning")
        wm = self.wm_learner.update(
            self.buffer.sample(self.cfg.horizon, self.cfg.wm_batch_size or self.cfg.batch_size)
            for _ in range(self.cfg.wm_batches)
        )
        policy_cfg = self.policy_learner.cfg
        policy = self.policy_learner.update(
            sample_start(self.buffer, self.cfg.history_length, policy_cfg.batch_size)
            for _ in range(policy_cfg.num_batches)
        )
        return {k: float(v) for k, v in (wm | policy).items()}
