# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
#
# Author: Kiwoong Park
"""BlinkMimic learning from Isaac articulation transitions, with torch imagination."""

from dataclasses import dataclass
import math
import time

import torch

from cyclo_lab.utils.episode_metrics import EpisodeMetrics
from .. import export
from cyclo_lab.manager_based.mimic._shared.motion_bridge import IsaacMotionBridge
from ..core.buffer import ReplayBuffer
from ..core.kinematics import Kinematics
from ..core.mirror import Mirror
from ..core.motion import GOAL_Q_OFFSET
from ..core.optim import MuonAdam
from ..core.mdp import MDP, RewardCfg, flatten_state, inv_dim
from ..core.networks import Critic, Policy, WorldModel
from ..core.learner import PolicyLearner, PolicyUpdateCfg, WorldModelLearner
from ..reference.motion import IsaacMotionLib
from ..core.rotations import quat_to_mat


@dataclass
class BlinkCfg:
    log_every: int = 50
    hist: int = 10
    goal_frames: int = 10
    wm_batch: int = 2048
    pi_batch: int = 2048
    utd: float = 0.4
    warmup_steps: int = 64
    eprobe_start: int = 1000
    eprobe_every: int = 125
    wm_window: int = 8
    horizon: int = 32
    enc_hidden: tuple = (2048, 1024, 512, 256)
    dec_hidden: tuple = (2048, 1024, 512, 256)
    wm_hidden: tuple = (1024, 1024, 1024)
    pi_ref_skip: bool = True
    wm_structured: bool = True
    wm_batches: int = 5
    critic_epochs: int = 5
    critic_minibatches: int = 4
    contact_reward: float = 0.25
    pi_ema: float = 0.995
    buffer_steps: int = 1024
    action_noise: float = 0.1
    pi_opt: str = "muon"
    wm_opt: str = "muon"
    lr: float = 4e-4
    gamma: float = 0.99
    lam: float = 0.95


class IsaacBlinkMimicRunner:
    actor_export = export
    family = "blinkmimic"
    active_encoders = ("motion",)

    def __init__(self, env, cfg):
        self.cfg, self.env = cfg, env
        self.bridge = bridge = IsaacMotionBridge(env)
        self.device = bridge.device
        self.metrics = EpisodeMetrics(bridge.env.num_envs, self.device, reward_scale=0.1)
        self.motion = IsaacMotionLib(bridge)
        self.sampler = None
        if hasattr(bridge.env, "encoder_bias"):
            from .reset_sampling import BlinkResetSampler

            command = bridge.env.command_manager.get_term("reference_trajectory")
            self.sampler = command.sampler = BlinkResetSampler(bridge.env, command, cfg.gamma)
        joints = bridge.num_actions
        default = bridge.robot.data.default_joint_pos.torch[0]
        self.xi = default.new_zeros(bridge.env.num_envs, 1 + joints + 3)
        reward_cfg = RewardCfg(term_height=bridge.anchor_pos_threshold, term_tilt=bridge.anchor_ori_threshold,
                               foot_contact_z=bridge.env.cfg.foot_contact_z)
        self.mdp = MDP(Kinematics(bridge), self.motion, default, self.xi.shape[-1], cfg=reward_cfg)
        self.mdp.feet = [bridge.fk.body_names.index(side + "_ankle_roll_link") for side in ("left", "right")]
        self.motion.set_ref_contact(self.mdp.feet)
        self.mirror = Mirror(
            bridge.robot,
            {"joint_offset": slice(1, joints + 1), "com": slice(joints + 1, joints + 4)},
            self.device,
        )
        scale, _ = bridge.action_transform()
        skip = (GOAL_Q_OFFSET, default.tolist(), scale.tolist()) if cfg.pi_ref_skip else None
        goal_dim = self.motion.goal_dim(cfg.goal_frames)
        self.actor = Policy(
            self.mdp.obs_dim,
            goal_dim,
            joints,
            hist=cfg.hist,
            enc_hidden=cfg.enc_hidden,
            dec_hidden=cfg.dec_hidden,
            ref_skip=skip,
        ).to(self.device)
        self.dynamics = WorldModel(
            joints,
            self.xi.shape[-1],
            hidden=cfg.wm_hidden,
            structured=cfg.wm_structured,
            dt=bridge.step_dt,
        ).to(self.device)
        self.critic = Critic(inv_dim(joints) + self.mdp.critic_extra + goal_dim).to(self.device)
        self.average = torch.optim.swa_utils.AveragedModel(
            self.actor, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(cfg.pi_ema), use_buffers=True
        )
        self.decay_start = None
        self.optimizers = {
            "dynamics": MuonAdam(
                self.dynamics, [self.dynamics.net], lr=cfg.lr, betas=(0.9, 0.999), fused=default.is_cuda
            ) if cfg.wm_opt == "muon" else torch.optim.Adam(
                self.dynamics.parameters(), lr=cfg.lr, fused=default.is_cuda
            ),
            "actor": MuonAdam(
                self.actor, [self.actor.enc, self.actor.dec], lr=4 * cfg.lr, fused=default.is_cuda
            ) if cfg.pi_opt == "muon" else torch.optim.Adam(
                self.actor.parameters(), lr=4 * cfg.lr, betas=(0.9, 0.95), fused=default.is_cuda
            ),
            "critic": torch.optim.Adam(self.critic.parameters(), lr=cfg.lr),
        }
        self.buffer = ReplayBuffer(
            cfg.buffer_steps, bridge.env.num_envs, self.device, joints, self.xi.shape[-1]
        )
        self.wm_learner = WorldModelLearner(
            self.dynamics, self.optimizers["dynamics"], self.normalized_context,
            num_joints=joints, window=cfg.wm_window, batch_size=cfg.wm_batch,
            mirror=self.mirror, mirror_fraction=0.5,
        )
        self.policy_learner = PolicyLearner(
            PolicyUpdateCfg(
                hist=cfg.hist, goal_frames=cfg.goal_frames, horizon=cfg.horizon,
                pi_batch=cfg.pi_batch, gamma=cfg.gamma, lam=cfg.lam, contact_reward=cfg.contact_reward,
                critic_epochs=cfg.critic_epochs, critic_minibatches=cfg.critic_minibatches,
                pi_clip_value=0.1, pi_ema=cfg.pi_ema,
            ),
            self.actor, self.critic, self.dynamics, self.optimizers["actor"], self.optimizers["critic"],
            self.mdp, self.motion, self.normalized_context, average=self.average,
            bounded_starts=True, keep_partial_minibatches=True, check_finite=True,
        )
        self.steps = torch.zeros(bridge.env.num_envs, device=self.device, dtype=torch.int32)
        self.previous = torch.zeros_like(bridge.robot.data.joint_pos.torch)
        self.history = None
        self.iteration = 0

    def normalized_context(self, raw):
        if not hasattr(self.bridge.env, "encoder_bias"):
            return raw
        return torch.cat(((raw[..., :1] - 0.75) / 0.45, raw[..., 1:-3] / 0.01, raw[..., -3:] / 0.05), -1)

    def state(self, x=None):
        if x is None:
            x = self.bridge.generalized_state()
        n = self.bridge.num_actions
        return (
            x[:, :3],
            quat_to_mat(x[:, 3:7]),
            x[:, 7 + n : 10 + n],
            x[:, 10 + n : 13 + n],
            x[:, 7 : 7 + n],
            x[:, 13 + n :],
        )

    @torch.no_grad()
    def step(self, training):
        state, f = self.state(), self.bridge.frame_ids()
        env = self.bridge.env
        if hasattr(env, "encoder_bias"):
            self.xi = torch.cat((env.tracking_friction, -env.encoder_bias, env.tracking_com_delta), -1)
        observation = self.mdp.obs(state, self.previous, self.xi, f)
        if training:
            observation = self.mdp.add_noise(observation)
        initial = observation[:, None].expand(-1, self.cfg.hist, -1)
        if self.history is None:
            self.history = initial
        else:
            shifted = torch.cat((self.history[:, 1:], observation[:, None]), dim=1)
            self.history = torch.where(self.steps[:, None, None] == 0, initial, shifted)
        goal = self.motion.goal(f, self.cfg.goal_frames)
        if training:
            self.actor.obs_norm.update(observation)
            self.actor.goal_norm.update(goal)
        policy = self.actor if training else self.average.module
        action = policy(self.history, goal)
        if training:
            action = (action + torch.randn_like(action) * self.cfg.action_noise).clamp(-20, 20)
        resets = self.bridge.reset_counts()
        _, reward, term, trunc, extras = self.env.step(action)
        done = term | trunc | (self.bridge.reset_counts() != resets)
        if training:
            after = self.bridge.generalized_state()
            captured = "final_obs" in extras and "tracking_state" in extras["final_obs"]
            if captured:
                after = torch.where((term | trunc)[:, None], extras["final_obs"]["tracking_state"], after)
            after = self.state(after)
            if self.sampler is not None:
                next_frame = (f + 1).minimum(self.motion.clip_end[self.motion.clip_id[f]])
                reward = self.mdp.reward(after, next_frame, action, self.previous)
                self.sampler.record(f, reward, term, done, self.steps)
            self.mean_step_reward = reward.mean() / 0.1
            if done.any():
                normalized = (self.metrics.returns + reward) / ((self.steps + 1) * 0.1)
                self.recent_tracking = float(normalized[done].mean())
            self.metrics.record(reward, done, terminated=term)
            wm = self.dynamics
            valid = (~done | captured) & torch.isfinite(flatten_state(after)).all(-1)
            if valid.any():
                wm.in_norm.update(
                    wm.inputs(state, action, self.previous, self.normalized_context(self.xi))[valid]
                )
                wm.out_norm.update(wm.targets(state, after)[valid])
                wm.inc_norm.update(wm.increments(state, after)[valid])
            self.buffer.add(
                dict(
                    s=flatten_state(state),
                    s1=flatten_state(after),
                    a=action,
                    a_prev=self.previous,
                    xi=self.xi,
                    f=f,
                    ep_step=self.steps,
                    done=done,
                    cut=done,
                    r=reward,
                )
            )
        self.steps = torch.where(done, 0, self.steps + 1)
        self.previous = torch.where(done[:, None], 0.0, action)

        return term, done

    def actor_checkpoint(self):
        return dict(
            pi=self.average.module.state_dict(),
            pi_live=self.actor.state_dict(),
            pi_avg=self.average.state_dict(),
        )

    def restore_actor(self, bundle, *, training=False):
        def weights(saved):
            if not training or not self.cfg.pi_ref_skip:
                return saved
            # Transfer training uses the current action transform. Playback and
            # export retain the learned policy's reference skip unchanged.
            saved = saved.copy()
            current = self.actor.state_dict()
            for name in ("ref_q0", "ref_alpha"):
                saved[name] = current[name]
            return saved

        self.actor.load_state_dict(weights(bundle.get("actor_live", bundle["actor"])))
        average = bundle.get("average_actor", bundle["actor"]) if training else bundle["actor"]
        self.average.module.load_state_dict(weights(average))
        self.average.n_averaged.fill_(bundle["ema_updates"])
        self.decay_start = bundle["decay_start"]

    def checkpoint_optimizers(self):
        return {name: [optimizer] for name, optimizer in self.optimizers.items()}

    def start_playback(self, encoder, velocity_command, motions):
        from cyclo_lab.manager_based.mimic._shared.play_viewer import TrackingPlayViewer

        return TrackingPlayViewer.create(self, motions)

    def play_step(self):
        self.step(False)

    def reset_playback_history(self):
        self.history = None
        self.steps.zero_()
        self.previous.zero_()

    def train_iteration(self, collection_steps=None):
        collection_start = time.perf_counter()
        if self.buffer.size == 0:
            for _ in range(max(self.cfg.warmup_steps, self.cfg.wm_window, self.cfg.hist)):
                self.step(True)
        steps = collection_steps
        if steps is None:
            rate = self.cfg.utd
            steps = int((self.iteration + 1) / rate) - int(self.iteration / rate)
        rewards = []
        for _ in range(steps):
            self.step(True)
            rewards.append(self.mean_step_reward)
        logs = self.metrics.scalars(reset=False)
        if "Train/mean_reward" in logs:
            logs["Train/mean_episode_return"] = logs.pop("Train/mean_reward")
        logs["Train/mean_reward"] = (
            torch.stack(rewards).mean().item() if rewards else self.mean_step_reward.item()
        )
        logs["Perf/collection_time"] = time.perf_counter() - collection_start
        return logs | self.update()

    def update(self):
        if self.decay_start is None and (
            self.iteration >= 1400 or getattr(self, "recent_tracking", 0) >= 0.75
        ):
            self.decay_start = self.iteration
        progress = (
            min(1.0, (self.iteration - self.decay_start) / 300) if self.decay_start is not None else 0.0
        )
        lr = self.cfg.lr * (0.03 + 0.97 * (1 + math.cos(math.pi * min(self.iteration / 250000, 1.0))) / 2)
        for name, optimizer in self.optimizers.items():
            for group in optimizer.param_groups:
                group["lr"] = lr * (4 - 3 * progress if name == "actor" else 1)
        # Collectors without terminal observations cannot train on the reset transition.
        wm_start = time.perf_counter()
        world_results = [
            self.wm_learner.update(self.buffer, exclude_cuts=self.sampler is None, check_finite=True)
            for _ in range(self.cfg.wm_batches)
        ]
        wm_loss, wm_error = torch.stack([torch.stack(result) for result in world_results]).mean(0).tolist()
        wm_time = time.perf_counter() - wm_start
        policy_start = time.perf_counter()
        policy_loss, value_loss, reward, grad_norm, survival = self.policy_learner.update(self.buffer)
        self.iteration += 1
        logs = {
            "Loss/world_model": wm_loss,
            f"BlinkMimic/world_model_error_{self.cfg.wm_window}_steps": wm_error,
            "Loss/policy": policy_loss.item(),
            "Loss/value_function": value_loss.item(),
            "Policy/gradient_norm": grad_norm.item(),
            "BlinkMimic/nonfinite_updates": self.policy_learner.nonfinite.item(),
            "BlinkMimic/imagined_reward": reward.item(),
            "BlinkMimic/imagined_survival": survival.item(),
            "Loss/learning_rate": lr,
            "Loss/policy_learning_rate": self.optimizers["actor"].param_groups[0]["lr"],
            "Policy/learning_rate_multiplier": self.optimizers["actor"].param_groups[0]["lr"] / lr,
        }
        logs["Perf/world_model_learning_time"] = wm_time
        logs["Perf/policy_learning_time"] = time.perf_counter() - policy_start
        logs["Perf/learning_time"] = wm_time + logs["Perf/policy_learning_time"]
        return logs
