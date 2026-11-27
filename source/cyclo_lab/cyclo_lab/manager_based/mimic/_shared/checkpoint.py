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

"""Select, save, validate and restore native motion tracker checkpoints."""

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
import torch

from .. import REPO_ROOT


def atomic_save(bundle, path):
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as file:
        temporary = Path(file.name)
    try:
        torch.save(bundle, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def checkpoint_motion_paths(paths):
    return [
        str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path)
        for path in map(Path, paths)
    ]


def save(runner, output, motions, *, task=None, task_motions=None):
    # Match each upstream convention: SuperMimic step index, BlinkMimic update count.
    is_supermimic = runner.family == "supermimic"
    step = runner.iteration - 1 if is_supermimic else runner.iteration
    bundle = dict(
        format="cyclo_lab.mimic.v2" if is_supermimic else "cyclo_lab.mimic.v1",
        task=task,
        task_motions=checkpoint_motion_paths(task_motions) if task_motions is not None else None,
        cfg=asdict(runner.cfg),
        contract={"family": runner.family},
        iteration=step,
        motions=checkpoint_motion_paths(motions),
        training_stats={"total_steps": runner.metrics.total_steps, "wall_time": runner.metrics.wall_time},
    )
    command = runner.bridge.env.command_manager.get_term("reference_trajectory")
    bundle["sampling"] = command.sampling_state()
    bundle.update(runner.actor_checkpoint())
    bundle["critic"] = runner.critic.state_dict()
    bundle["dynamics"] = runner.dynamics.state_dict()
    if runner.family == "supermimic":
        bundle["sim_context"] = runner.bridge.sim_context.state_dict()
    snapshot = None
    if step % 1000 == 0:
        snapshot = output / f"model_{step}.pt"
        atomic_save(bundle, snapshot)
        snapshots = sorted(
            (int(path.stem[6:]), path) for path in output.glob("model_*.pt") if path.stem[6:].isdecimal()
        )
        for iteration, path in snapshots[:-5]:
            if iteration % 10000:
                path.unlink()
    bundle["optimizers"] = {k: [opt.state_dict() for opt in opts] for k, opts in runner.checkpoint_optimizers().items()}
    atomic_save(bundle, output / "last.pt")
    return snapshot


def load(runner, bundle, *, training, world_only=False):
    weights = bundle["dynamics"]
    if runner.family == "supermimic":
        if "cfg" not in bundle and not hasattr(runner.bridge.env, "encoder_bias"):
            raise ValueError("Native world-model transfer requires the friction, COM and encoder-bias context")
        if "sim_context" in bundle:
            context = bundle["sim_context"]
            if not all(torch.isfinite(context[key]).all() for key in ("mean", "scale")) or not (
                context["scale"] > 0
            ).all():
                raise ValueError("World-model context normalization must be finite with positive scales")
            try:
                runner.bridge.sim_context.load_state_dict(context)
            except RuntimeError as exc:
                raise ValueError(f"World-model context is incompatible with this task: {exc}") from exc
        # These buffers convert actions into joint targets in the destination environment.
        weights = weights.copy()
        for name in ("action_scale", "action_offset"):
            if name in weights:
                weights[name] = getattr(runner.dynamics, name)
    try:
        runner.dynamics.load_state_dict(weights)
    except RuntimeError as exc:
        raise ValueError(f"World-model weights are incompatible with this task: {exc}") from exc
    if world_only:
        return
    if training and "sampling" in bundle:
        runner.bridge.env.command_manager.get_term("reference_trajectory").restore_sampling(
            bundle["sampling"]
        )
    runner.restore_actor(bundle, training=training)
    runner.critic.load_state_dict(bundle["critic"])
    if training and "optimizers" not in bundle:
        import warnings

        warnings.warn("Model-only checkpoint: optimizer state starts fresh", stacklevel=2)
    if training and "optimizers" in bundle:
        for name, opts in runner.checkpoint_optimizers().items():
            for opt, state in zip(opts, bundle["optimizers"][name], strict=True):
                opt.load_state_dict(state)
    runner.iteration = bundle["iteration"]
    if runner.family == "supermimic" and bundle.get("format") == "cyclo_lab.mimic.v2":
        runner.iteration += 1
    if training:
        stats = bundle.get("training_stats", {})
        runner.metrics.total_steps = stats.get("total_steps", 0)
        runner.metrics.wall_time = stats.get("wall_time", 0.0)


def training_run_order(directory):
    try:
        started = datetime.strptime(directory.name[:19], "%Y-%m-%d_%H-%M-%S").timestamp()
    except ValueError:
        snapshots = [p for p in directory.glob("*.pt") if p.is_file() and not p.is_symlink()
                     and (p.name == "last.pt" or p.stem.startswith("model_") and p.stem[6:].isdecimal())]
        started = max((p.stat().st_mtime for p in snapshots), default=float("-inf"))
    return started, directory.name


def find_checkpoint(root, task, path, run_pattern=None, checkpoint_pattern=None):
    from cyclo_lab.utils.training_resume import ResumeOptions, select_resume_checkpoint

    if path is not None:
        selected = Path(path).expanduser()
        selected = selected / "last.pt" if selected.is_dir() else selected
        if not selected.is_file():
            raise FileNotFoundError(selected)
        return selected
    return select_resume_checkpoint(
        root / task,
        ResumeOptions(True, run_pattern or ".*", checkpoint_pattern or "last.pt"),
        skip_discovered_symlinks=True,
        run_sort_key=training_run_order,
    )


def load_checkpoint(action, args, family, task, root):
    if args.world_model_checkpoint and (args.checkpoint or args.resume):
        raise ValueError("Choose a full checkpoint or a world-model checkpoint")
    if action == "train" and not (args.checkpoint or args.resume or args.world_model_checkpoint):
        return None, None
    selected = args.checkpoint or args.world_model_checkpoint
    if args.world_model_checkpoint and selected.expanduser().is_dir():
        directory = selected.expanduser()
        selected = directory / ("last.pt" if (directory / "last.pt").is_file() else "world_model.pt")
    path = find_checkpoint(
        root,
        task,
        selected,
        args.run_pattern,
        args.checkpoint_pattern,
    )
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    is_cyclo_checkpoint = bundle.get("format") in ("cyclo_lab.mimic.v1", "cyclo_lab.mimic.v2")
    if family == "supermimic" and not is_cyclo_checkpoint:
        if not args.world_model_checkpoint:
            raise ValueError("A native world_model.pt only initializes the world model; use --world-model-checkpoint")
        if not all(isinstance(bundle.get(key), dict) for key in ("dynamics", "sim_context")):
            raise ValueError("Expected a SuperMimic world_model.pt containing dynamics and sim_context")
        return path, bundle
    if not is_cyclo_checkpoint or bundle["contract"]["family"] != family:
        raise ValueError("Checkpoint algorithm or format does not match this Isaac Lab task")
    return path, bundle


def restore_runner(runner, bundle, *, training, world_only, iterations):
    if bundle:
        load(runner, bundle, training=training, world_only=world_only)
        if training and iterations <= runner.iteration:
            raise ValueError("--iterations must exceed the saved iteration")
