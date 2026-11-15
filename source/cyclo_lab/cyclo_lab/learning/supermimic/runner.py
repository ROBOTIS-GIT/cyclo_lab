# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Isaac collection loop around original SuperMimic WM and imagined BPTT learners."""

from dataclasses import dataclass, asdict
from pathlib import Path
import torch
import hashlib
from tensordict import TensorDict
from supermimic.models import Dynamics, MultiEncoderAE, build_mlp
from supermimic.representations import create_representation
from supermimic.replay_buffer import ReplayBuffer, FieldSpec
from supermimic.tasks.task_spec import make_motion_task, build_actor_obs_td
from supermimic.envs.multi_task_world_model_env import MultiTaskWorldModelEnv
from supermimic.learners.wm_learner import WorldModelLearner, WorldModelLearnerCfg
from supermimic.learners.policy_learner import (
    ActorCriticPolicyLearner,
    ActorCriticPolicyLearnerCfg,
)
from .bridge import IsaacBridge


@dataclass
class RunnerCfg:
    hidden: tuple[int, ...] = (256, 256)
    latent: int = 32
    batch_size: int = 64
    horizon: int = 8
    policy_horizon: int = 16
    buffer_size: int = 1024
    collection_steps: int = 16
    noise: float = 0.1
    eager: bool = False


class IsaacSuperMimicRunner:
    def __init__(self, env, cfg: RunnerCfg):
        if cfg.collection_steps < 1 or cfg.batch_size < 1:
            raise ValueError("Collection steps and batch size must be positive")
        if cfg.horizon < 2 or cfg.batch_size * cfg.policy_horizon < 4:
            raise ValueError(
                "WM needs >=2 frames; actor-critic needs >=4 rollout samples"
            )
        if cfg.buffer_size < cfg.horizon:
            raise ValueError("Replay buffer smaller than WM horizon")
        self.cfg, self.env = cfg, env
        self.bridge = IsaacBridge(env)
        b = self.bridge
        self.device = b.device
        self.iteration = 0
        self.collected_steps = 0
        self.rep = create_representation(
            "gencoord",
            59,
            23,
            b.step_dt,
            env.unwrapped.cfg.decimation,
            b.robot.data.default_joint_pos.torch[0].clone(),
            "rep_state",
        ).to(self.device)
        self.rep._soft_joint_pos_limits = b.robot.data.soft_joint_pos_limits.torch[
            0
        ].clone()
        self.rep.fk = b.fk
        self.scale, self.offset = b.action_transform()
        self.dynamics = Dynamics(
            self.rep, 23, 1, self.scale, self.offset, cfg.hidden
        ).to(self.device)
        goal, _ = self.rep.compute_wm_goal(b.motion, b.frame_ids(), b.state()[:, :7])
        self.specs = [make_motion_task(self.rep, goal.shape[-1])]
        self.actor = MultiEncoderAE(
            self.rep.rep_state_dim,
            23,
            self.specs,
            {"motion": 1.0},
            cfg.latent,
            cfg.hidden,
            0.0,
            1,
        ).to(self.device)
        self.critic = build_mlp(
            self.rep.rep_state_dim + goal.shape[-1] + 1, cfg.hidden, 1
        ).to(self.device)
        self.buffer = ReplayBuffer(
            {
                "state": FieldSpec(59),
                "action": FieldSpec(23),
                "prev_action": FieldSpec(23),
                "sim_ctx": FieldSpec(1),
                "motion_time": FieldSpec(dtype=torch.long),
                "terminated": FieldSpec(dtype=torch.bool),
                "truncated": FieldSpec(dtype=torch.bool),
            },
            cfg.buffer_size,
            env.unwrapped.num_envs,
            self.device,
        )
        self.wm_env = MultiTaskWorldModelEnv(
            self.dynamics,
            self.rep,
            b,
            cfg.batch_size,
            self.buffer,
            self.specs,
            1,
            0.05,
            self.device,
        )
        self.wm_learner = WorldModelLearner(
            self.dynamics,
            self.rep,
            WorldModelLearnerCfg(
                num_batches=1, batch_size=cfg.batch_size, horizon=cfg.horizon
            ),
            self.device,
        )
        self.policy_learner = ActorCriticPolicyLearner(
            self.actor,
            self.critic,
            self.wm_env,
            ActorCriticPolicyLearnerCfg(
                num_batches=1, batch_size=cfg.batch_size, horizon=cfg.policy_horizon
            ),
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
        self.previous_action = torch.zeros(
            (env.unwrapped.num_envs, 23), device=self.device
        )

    def actor_obs(self, state, frames):
        aux = self.rep.extract_aux_state(state)
        _, goal = self.rep.compute_wm_goal(self.bridge.motion, frames, aux)
        sensor = self.rep.represent(state)
        return build_actor_obs_td({"motion": goal}, sensor[:, None])

    def collect(self, steps):
        collected = []
        for _ in range(steps):
            # Snapshot BEFORE env.step: that call can reset and advance motion in place.
            state, frames = self.bridge.state(), self.bridge.frame_ids()
            with torch.no_grad():
                self.actor.eval()
                obs = self.actor_obs(state, frames)
                action = (
                    self.actor(obs)
                    + torch.randn_like(self.previous_action) * self.cfg.noise
                ).clamp(-10, 10)
                _, _, terminated, truncated, _ = self.env.step(action)
                transition = TensorDict(
                    dict(
                        state=state,
                        action=action.clone(),
                        prev_action=self.previous_action.clone(),
                        motion_time=frames,
                        sim_ctx=self.bridge.context(),
                        terminated=terminated.clone(),
                        truncated=truncated.clone(),
                    ),
                    batch_size=[len(state)],
                    device=self.device,
                )
                self.buffer.insert(transition)
                self.previous_action = action.clone()
                self.previous_action[terminated | truncated] = 0
                collected.append((state, action, obs))
                self.collected_steps += 1
        self.wm_learner.update_normalization(
            torch.cat([x[0] for x in collected]), torch.cat([x[1] for x in collected])
        )
        self.actor.train()
        for _, _, obs in collected:
            self.actor.update_normalization(obs)

    def update(self):
        if self.buffer.num_transitions < self.cfg.horizon:
            raise RuntimeError("Collect enough Isaac transitions before learning")
        wm_before = [p.detach().clone() for p in self.dynamics.parameters()]
        actor_before = [p.detach().clone() for p in self.actor.parameters()]
        wm = self.wm_learner.update(
            iter([self.buffer.sample(self.cfg.horizon, self.cfg.batch_size)])
        )
        policy = self.policy_learner.update()
        logs = {k: float(v) for k, v in (wm | policy).items()}
        if not all(torch.isfinite(torch.tensor(v)) for v in logs.values()):
            raise RuntimeError(f"Nonfinite update metrics: {logs}")
        logs["wm_parameters_changed"] = any(
            not torch.equal(a, b) for a, b in zip(wm_before, self.dynamics.parameters())
        )
        logs["actor_parameters_changed"] = any(
            not torch.equal(a, b) for a, b in zip(actor_before, self.actor.parameters())
        )
        if not logs["wm_parameters_changed"] or not logs["actor_parameters_changed"]:
            raise RuntimeError("Learner update did not change model parameters")
        self.iteration += 1
        return logs

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            dict(
                format="cyclo-supermimic-isaac-v1",
                task=self.env.spec.id,
                motion_file=str(
                    Path(
                        self.env.unwrapped.cfg.commands.reference_trajectory.trajectory_file
                    ).resolve()
                ),
                cfg=asdict(self.cfg),
                iteration=self.iteration,
                joint_names=list(self.bridge.robot.joint_names),
                dt=self.bridge.step_dt,
                motion_sha256=hashlib.sha256(
                    Path(
                        self.env.unwrapped.cfg.commands.reference_trajectory.trajectory_file
                    ).read_bytes()
                ).hexdigest(),
                action_scale=self.scale,
                action_offset=self.offset,
                dynamics=self.dynamics.state_dict(),
                actor=self.actor.state_dict(),
                critic=self.critic.state_dict(),
                wm_optimizers=[o.state_dict() for o in self.wm_learner.optimizers],
                actor_optimizers=[
                    o.state_dict() for o in self.policy_learner.actor_optimizers
                ],
                critic_optimizers=[
                    o.state_dict() for o in self.policy_learner.critic_optimizers
                ],
            ),
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("format") != "cyclo-supermimic-isaac-v1":
            raise ValueError(
                "Expected Isaac SuperMimic checkpoint, not a PPO or mjlab checkpoint"
            )
        if (
            checkpoint["joint_names"] != list(self.bridge.robot.joint_names)
            or checkpoint["dt"] != self.bridge.step_dt
        ):
            raise ValueError("Checkpoint joint order or policy timestep differs")
        if not torch.equal(checkpoint["action_scale"], self.scale) or not torch.equal(
            checkpoint["action_offset"], self.offset
        ):
            raise ValueError("Checkpoint actuation differs")
        for name in ("dynamics", "actor", "critic"):
            getattr(self, name).load_state_dict(checkpoint[name])
        for name, optimizers in [
            ("wm", self.wm_learner.optimizers),
            ("actor", self.policy_learner.actor_optimizers),
            ("critic", self.policy_learner.critic_optimizers),
        ]:
            for optimizer, state in zip(
                optimizers, checkpoint[name + "_optimizers"], strict=True
            ):
                optimizer.load_state_dict(state)
        self.iteration = checkpoint["iteration"]
        if not self.cfg.eager:
            from supermimic.learners.learner_utils import CapturedUpdate, graph_state

            self.wm_learner._captured_update = CapturedUpdate(
                self.wm_learner._update_from,
                lambda: graph_state(
                    [self.dynamics, self.rep], self.wm_learner.optimizers
                ),
            )
            self.policy_learner._captured_update = CapturedUpdate(
                self.policy_learner._update_from, self.policy_learner._graph_state
            )
