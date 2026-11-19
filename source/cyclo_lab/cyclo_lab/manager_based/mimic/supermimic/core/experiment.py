# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""SuperMimic encoder, reward and checkpoint settings for an Isaac tracking run."""

from .runner import IsaacSuperMimicRunner as Runner

__all__ = ["Runner", "configure_learner", "configure_environment", "training_probe", "evaluate"]

training_probe = None
evaluate = None


def configure_environment(cfg, action, args):
    from ..config.k1_rev1.base_env_cfg import validate_imagined_rewards

    validate_imagined_rewards(cfg)


def configure_learner(settings, args, bundle):
    fraction = args.velocity_fraction
    if fraction is not None:
        if not 0 <= fraction <= 1:
            raise ValueError("--velocity-fraction must be in [0, 1]")
        settings.velocity_fraction = fraction
    if args.batch_size is not None:
        settings.wm_batch_size = args.batch_size
    if args.hidden:
        settings.critic_hidden = tuple(args.hidden)
    if bundle:
        saved = bundle["cfg"]
        if settings.representation != saved["representation"]:
            raise ValueError("Checkpoint representation differs from the selected task")
        if not args.world_model_checkpoint:
            settings = type(settings)(**{k: v for k, v in saved.items() if k not in ("noise", "verify_updates")})
            if fraction is not None and fraction != settings.velocity_fraction:
                raise ValueError("Full resume preserves the encoder mix; use --world-model-checkpoint to change it")
    settings.eager = args.eager
    return settings
