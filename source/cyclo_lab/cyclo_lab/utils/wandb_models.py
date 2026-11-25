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

"""Opt-in online publication of local training checkpoints."""

from contextlib import contextmanager
import os


def model_upload_requested():
    return os.environ.get("CYCLO_WANDB_UPLOAD_MODELS") == "1"


@contextmanager
def model_upload_run(directory, name):
    if not model_upload_requested() or os.environ.get(
        "WANDB_MODE", "online"
    ).lower() in {"offline", "dryrun", "disabled"}:
        yield None
        return

    try:
        import wandb
    except ModuleNotFoundError as error:
        if error.name != "wandb":
            raise
        raise RuntimeError(
            "CYCLO_WANDB_UPLOAD_MODELS=1 requires the wandb package"
        ) from error

    with wandb.init(dir=str(directory), name=name) as run:
        yield run if run.settings.mode == "online" else None


def upload_checkpoint(run, path):
    run.log_artifact(str(path), name=f"model-{run.id}", type="model")
