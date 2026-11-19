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
