# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""BlinkMimic settings and evaluation for an Isaac tracking run."""

from .evaluation import run_training_probe_if_due as training_probe, write_evaluation_report as evaluate
from .runner import IsaacBlinkMimicRunner as Runner

__all__ = ["Runner", "configure_learner", "configure_environment", "training_probe", "evaluate"]


def configure_environment(cfg, action, args):
    if action != "train" and cfg.events.physics_material:
        cfg.events.physics_material.params["evaluation"] = True
        cfg.events.joint_home_offset_noise = None
        ref = cfg.commands.reference_trajectory
        ref.reset_pose_noise = {}
        ref.reset_velocity_noise = {}
        ref.reset_joint_position_noise = (0.0, 0.0)
    if args.evaluate or args.probe_report:
        cfg.terminations.time_out = None


def configure_learner(settings, args, bundle):
    if args.velocity_fraction not in (None, 0):
        raise ValueError("BlinkMimic supports only the motion encoder")
    if bundle:
        if not args.world_model_checkpoint:
            return type(settings)(**bundle["cfg"])
        settings.structured_dynamics = bundle["cfg"]["structured_dynamics"]
    return settings
