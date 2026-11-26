# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
#
# Author: Kiwoong Park
"""SuperMimic encoder, reward and checkpoint settings for an Isaac tracking run."""

from copy import deepcopy

from .training.runner import (
    IsaacSuperMimicRunner as Runner, RunnerCfg,
    WorldModelLearnerCfg, ActorCriticPolicyLearnerCfg,
)

__all__ = ["Runner", "restore_config", "configure_learner", "configure_environment", "training_probe", "evaluate"]

training_probe = None
evaluate = None


def configure_environment(cfg, action, args):
    from .config.k1_rev1.base_env_cfg import validate_imagined_rewards

    validate_imagined_rewards(cfg)


def restore_config(saved):
    """Convert the former flat Cyclo config to the native learner structure."""
    saved = deepcopy(saved)
    if "hidden" in saved:
        hidden = saved.pop("hidden")
        saved["actor_hidden_dims"] = hidden
        saved["dynamics_hidden_dims"] = saved.pop("world_hidden") or hidden
        saved["critic_hidden_dims"] = saved.pop("critic_hidden") or hidden
        saved["latent_dim"] = saved.pop("latent")
        batch = saved.pop("batch_size")
        saved["wm"] = dict(
            batch_size=saved.pop("wm_batch_size") or batch,
            num_batches=saved.pop("wm_batches"), horizon=saved.pop("horizon"),
        )
        saved["policy"] = dict(batch_size=batch, horizon=saved.pop("policy_horizon"), num_batches=1)
        fraction = saved.pop("velocity_fraction")
        saved["slot_mix"] = {"motion": 1.0 - fraction, "velocity": fraction}
        saved["latent_bc_weight"] = 0.1 if fraction else 0.0
    for name in ("collection_steps", "noise", "verify_updates"):
        saved.pop(name, None)
    saved["wm"] = WorldModelLearnerCfg(**saved["wm"])
    saved["policy"] = ActorCriticPolicyLearnerCfg(**saved["policy"])
    return RunnerCfg(**saved)


def configure_learner(settings, args, bundle):
    if bundle and "cfg" not in bundle:
        # Native world_model.pt has weights, but no runner configuration.
        layers = sorted(
            (int(name.split(".")[1]), weight)
            for name, weight in bundle["dynamics"].items()
            if name.startswith("mlp.") and name.endswith(".weight") and weight.ndim == 2
        )
        if not layers:
            raise ValueError("World-model checkpoint has no Dynamics MLP weights")
        settings.dynamics_hidden_dims = tuple(weight.shape[0] for _, weight in layers[:-1])
    elif bundle:
        saved = restore_config(bundle["cfg"])
        if settings.representation != saved.representation:
            raise ValueError("Checkpoint representation differs from the selected task")
        if args.world_model_checkpoint:
            settings.dynamics_hidden_dims = saved.dynamics_hidden_dims
        else:
            if settings.slot_mix != saved.slot_mix:
                raise ValueError("Full resume preserves the encoder mix; use --world-model-checkpoint to change it")
    if args.eager is not None:
        settings.eager = args.eager
    return settings
