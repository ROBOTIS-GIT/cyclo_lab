# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Isaac-only variants: preserve existing BeyondMimic task definitions."""

from isaaclab.utils import configclass
from cyclo_lab.manager_based.mimic.config.k1_rev1.dance1_env_cfg import (
    K1Rev1EnvCfg as Dance1,
)
from cyclo_lab.manager_based.mimic.config.k1_rev1.dance2_env_cfg import (
    K1Rev1EnvCfg as Dance2,
)


def configure_supermimic(cfg):
    cfg.commands.reference_trajectory.anchor_body_name = "pelvis"
    cfg.commands.reference_trajectory.debug_vis = False
    # Constant dynamics context in the first Isaac port: no hidden actuator/COM DR.
    for name in (
        "physics_material",
        "joint_home_offset_noise",
        "torso_com_offset_noise",
        "push_robot",
    ):
        setattr(cfg.events, name, None)
    cfg.rewards.joint_acc = None
    cfg.rewards.joint_torque = None
    cfg.rewards.undesired_contacts = None
    cfg.rewards.joint_limit.weight = -2.0
    cfg.terminations.end_body_pos = None
    cfg.export_sim2real_cfg = False


@configclass
class SuperMimicDance1EnvCfg(Dance1):
    def __post_init__(self):
        super().__post_init__()
        configure_supermimic(self)


@configclass
class SuperMimicDance2EnvCfg(Dance2):
    def __post_init__(self):
        super().__post_init__()
        configure_supermimic(self)
