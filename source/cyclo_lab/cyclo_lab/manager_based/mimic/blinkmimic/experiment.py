# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
#
# Author: Kiwoong Park
"""BlinkMimic settings and evaluation for an Isaac tracking run."""

from .evaluation import run_training_probe_if_due as training_probe, write_evaluation_report as evaluate
from .training.runner import BlinkCfg, IsaacBlinkMimicRunner as Runner

__all__ = ["Runner", "restore_config", "configure_learner", "configure_environment", "training_probe", "evaluate"]


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


def restore_config(saved):
    return BlinkCfg(**saved)


def configure_learner(settings, args, bundle):
    for name in ("pi_opt", "wm_opt"):
        if getattr(settings, name) not in ("adam", "muon"):
            raise ValueError(f"agent.{name} must be 'adam' or 'muon'")
    if bundle and args.world_model_checkpoint:
        settings.wm_hidden = restore_config(bundle["cfg"]).wm_hidden
    if min(settings.wm_batch, settings.pi_batch, settings.wm_batches, settings.horizon, settings.hist) < 1:
        raise ValueError("BlinkMimic batch sizes, horizons and history must be positive")
    if settings.utd <= 0 or not 2 <= settings.wm_window <= settings.buffer_steps:
        raise ValueError("Invalid BlinkMimic update rate or world-model window")
    return settings
