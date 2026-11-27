# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
    """Rebuild the nested learner dataclasses from recorded settings."""
    saved = deepcopy(saved)
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
