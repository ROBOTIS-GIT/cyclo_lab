# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
#
# Author: Kiwoong Park
"""BlinkMimic settings and evaluation for an Isaac tracking run."""

from copy import deepcopy

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
    """Read old Cyclo checkpoint field names without retaining CLI aliases."""
    saved = deepcopy(saved)
    if "policy_horizon" in saved:
        saved["wm_window"] = saved.pop("horizon")
        saved["horizon"] = saved.pop("policy_horizon")
    for old, new in {
        "history_length": "hist", "updates_per_step": "utd",
        "probe_start": "eprobe_start", "probe_interval": "eprobe_every",
        "world_hidden": "wm_hidden", "reference_skip": "pi_ref_skip",
        "ema": "pi_ema", "buffer_size": "buffer_steps", "noise": "action_noise",
        "structured_dynamics": "wm_structured",
    }.items():
        if old in saved:
            saved[new] = saved.pop(old)
    if "batch_size" in saved:
        saved["wm_batch"] = saved["pi_batch"] = saved.pop("batch_size")
    if "hidden" in saved:
        saved["enc_hidden"] = saved["dec_hidden"] = saved.pop("hidden")
    saved.pop("verify_updates", None)
    return BlinkCfg(**saved)


def configure_learner(settings, args, bundle):
    if bundle and args.world_model_checkpoint:
        settings.wm_hidden = restore_config(bundle["cfg"]).wm_hidden
    if min(settings.wm_batch, settings.pi_batch, settings.wm_batches, settings.horizon, settings.hist) < 1:
        raise ValueError("BlinkMimic batch sizes, horizons and history must be positive")
    if settings.utd <= 0 or not 2 <= settings.wm_window <= settings.buffer_steps:
        raise ValueError("Invalid BlinkMimic update rate or world-model window")
    return settings
