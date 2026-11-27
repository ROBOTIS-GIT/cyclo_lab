# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
#
# Author: Kiwoong Park

from isaaclab.utils import configclass
from cyclo_lab.manager_based.mimic._shared.motion_library_env_cfg import K1MotionLibraryEnvCfg, TrackingRewardsCfg
from ...mdp.rewards import heading_velocity_tracking


@configclass
class SuperMimicRewardsCfg(TrackingRewardsCfg):
    def __post_init__(self):
        self.reference_body_linear_velocity.func = heading_velocity_tracking
        self.reference_body_angular_velocity.func = heading_velocity_tracking
        self.reference_body_angular_velocity.params["angular"] = True


@configclass
class SuperMimicLibraryEnvCfg(K1MotionLibraryEnvCfg):
    rewards: SuperMimicRewardsCfg = SuperMimicRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.init_state.joint_pos[".*_elbow_joint"] = 0.95
        self.events.physics_material.params["feet_only"] = True
        self.events.joint_home_offset_noise = None
        solver = self.sim.physics.newton_mjwarp.solver_cfg
        solver.ccd_iterations = 50
        solver.tolerance = 1e-8
        self.commands.reference_trajectory.targets_next_frame = True


def validate_imagined_rewards(cfg):
    from cyclo_lab.manager_based.mimic._shared.tracking_contract import active_terms

    if active_terms(cfg.rewards) != active_terms(SuperMimicRewardsCfg()):
        raise ValueError("Reward override differs from the SuperMimic imagined objective")
