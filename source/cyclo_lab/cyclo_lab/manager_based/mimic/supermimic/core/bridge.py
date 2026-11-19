"""SuperMimic representation and world-model views over the Isaac collector."""

from types import SimpleNamespace
import torch
from cyclo_lab.manager_based.mimic.motion_bridge import BODIES, URDF, IsaacMotionBridge, wxyz


class IsaacBridge(IsaacMotionBridge):
    def __init__(self, env, representation="gencoord"):
        super().__init__(env)
        self.representation = representation
        self.tracked_ids = [self.robot.body_names.index(name) for name in BODIES]
        self.full_state_dim = 59 if representation == "gencoord" else 13 * len(BODIES)

    def world_model_facade(self):
        """Configuration-only protocol consumed by the unchanged official WM env.

        This exposes motion and scalar objective settings, never simulator IDs or
        a surrogate physics model. All collection continues through the real env.
        """
        motion_command = SimpleNamespace(motion=self.motion, motion_anchor_body_index=self.anchor_index)

        def get_term(name):
            if name != "motion":
                raise KeyError(name)
            return motion_command

        rewards = {
            k: v for k, v in vars(self.env.cfg.rewards).items() if not k.startswith("_") and v is not None
        }
        return SimpleNamespace(
            num_actions=self.num_actions,
            cfg=SimpleNamespace(
                rewards=rewards,
                terminations={
                    "anchor_pos": SimpleNamespace(params={"threshold": self.anchor_pos_threshold}),
                    "anchor_ori": SimpleNamespace(params={"threshold": self.anchor_ori_threshold}),
                },
            ),
            unwrapped=SimpleNamespace(
                step_dt=self.step_dt,
                command_manager=SimpleNamespace(get_term=get_term),
            ),
        )

    def state(self):
        if self.representation == "gencoord":
            return self.generalized_state()
        d = self.robot.data
        ids = self.tracked_ids
        # Official field-major layout; all quantities are link-origin world states.
        fields = (
            d.body_link_pos_w.torch[:, ids] - self.env.scene.env_origins[:, None],
            wxyz(d.body_link_quat_w.torch[:, ids]),
            self.tracked_linear_velocity(),
            d.body_link_ang_vel_w.torch[:, ids],
        )
        return torch.cat([v.flatten(-2) for v in fields], -1).clone()

    def tracked_linear_velocity(self):
        data = self.robot.data
        velocities = data.body_link_lin_vel_w.torch[:, self.tracked_ids]
        if type(self.env.cfg.sim.physics).__module__.startswith("isaaclab_physx."):
            # After a state write, PhysX updates the torso COM velocity on its first
            # physics tick. Until then use its waist joint pivot, fixed on the pelvis.
            torso = BODIES.index("torso_link")
            arm = data.body_link_pos_w.torch[:, self.tracked_ids[torso]] - data.root_link_pos_w.torch
            pivot_velocity = data.root_link_lin_vel_w.torch + torch.linalg.cross(
                data.root_link_ang_vel_w.torch, arm
            )
            reset = self.env.episode_length_buf[:, None] == 0
            velocities[:, torso] = torch.where(reset, pivot_velocity, velocities[:, torso])
        return velocities

    def context(self):
        if not hasattr(self.env, "encoder_bias"):
            return torch.zeros((self.env.num_envs, 1), device=self.device)
        return torch.cat(
            (
                (self.env.tracking_friction - 0.75) / 0.45,
                self.env.tracking_com / self.env.tracking_com.new_tensor([0.025, 0.05, 0.05]),
                self.env.encoder_bias / 0.01,
            ),
            dim=-1,
        )
