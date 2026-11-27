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

from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
import torch

from .. import REPO_ROOT


SUPERMIMIC_FORMAT = "cyclo.supermimic.checkpoint.v1"
BLINKMIMIC_OPTIMIZERS = {"dynamics": "opt_wm", "actor": "opt_pi", "critic": "opt_v"}


def supermimic_manifest(runner, task, cfg):
    from .motion_bridge import BODIES
    from .tracking_contract import _plain

    bridge = runner.bridge
    names = list(bridge.robot.joint_names)
    return dict(
        schema_version=2,
        task_id=task,
        representation=runner.cfg.representation,
        actor_obs=runner.cfg.actor_obs,
        runner_cfg=cfg,
        environment={
            "commands": {"motion": {"body_names": list(BODIES), "anchor_body_name": "pelvis"}},
            # Record the real Isaac configuration, not invented MJLab callables.
            "observations": _plain(bridge.env.cfg.observations),
        },
        robot={"joint_order": names, "action_joint_order": names},
        q0=bridge.robot.data.default_joint_pos.torch[0].tolist(),
        qdot0=bridge.robot.data.default_joint_vel.torch[0].tolist(),
    )


def add_optimizers(saved, states, family):
    if family == "supermimic":
        saved["optimizers"] = {("world_model" if key == "dynamics" else key): value
                               for key, value in states.items()}
    else:
        for name, key in BLINKMIMIC_OPTIMIZERS.items():
            (saved[key],) = states[name]


def decode(saved, family):
    """Read each algorithm's archive into the shared runner inputs once."""
    if family == "supermimic":
        if saved.get("format") != SUPERMIMIC_FORMAT:
            if saved.get("format") is None and all(
                isinstance(saved.get(key), dict) for key in ("dynamics", "sim_context")
            ):
                return saved
            raise ValueError("Expected a SuperMimic checkpoint or world-model bundle")
        manifest = saved["manifest"]
        from ..supermimic.training.runner import RunnerCfg

        cfg = {f.name: manifest["runner_cfg"][f.name] for f in fields(RunnerCfg)
               if f.name in manifest["runner_cfg"]}
        result = dict(
            cfg=cfg, iteration=saved["step"] + 1, actor=saved["actor"],
            **saved["world_model"],
        )
        if saved.get("critic") is not None:
            result["critic"] = saved["critic"]
        if saved.get("optimizers") is not None:
            result["optimizers"] = {("dynamics" if key == "world_model" else key): value
                                    for key, value in saved["optimizers"].items()}
        result["training_stats"] = saved.get("training_stats", {})
    else:
        if not all(key in saved for key in ("args", "pi", "iteration")):
            raise ValueError("Expected a BlinkMimic checkpoint containing args, pi and iteration")
        from ..blinkmimic.training.runner import BlinkCfg

        cfg = {f.name: saved["args"][f.name] for f in fields(BlinkCfg) if f.name in saved["args"]}
        for key in ("enc_hidden", "dec_hidden", "wm_hidden"):
            if isinstance(cfg.get(key), str):
                cfg[key] = tuple(map(int, cfg[key].split(",")))
        average = saved.get("pi_avg")
        result = dict(
            cfg=cfg, iteration=saved["iteration"], actor=saved["pi"],
            actor_live=saved.get("pi_live", saved["pi"]),
            ema_updates=int(average["n_averaged"]) if average is not None else 1,
            decay_start=(saved.get("ctrl") or {}).get("decay_at"),
            training_stats={"total_steps": saved.get("sim_steps", 0), "wall_time": saved.get("wall_time", 0.0)},
        )
        if average is not None:
            result["average_actor"] = {key.removeprefix("module."): value for key, value in average.items()
                                       if key.startswith("module.")}
        for source, target in (("wm", "dynamics"), ("critic", "critic")):
            if source in saved:
                result[target] = saved[source]
        keys = set(BLINKMIMIC_OPTIMIZERS.values())
        if keys.intersection(saved):
            if not keys.issubset(saved):
                raise ValueError("Incomplete optimizer state in BlinkMimic checkpoint")
            result["optimizers"] = {name: [saved[key]] for name, key in BLINKMIMIC_OPTIMIZERS.items()}
    result.update(saved.get("cyclo_lab", {}))
    return result


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
    local = dict(
        task=task,
        task_motions=checkpoint_motion_paths(task_motions) if task_motions is not None else None,
        motions=checkpoint_motion_paths(motions),
    )
    command = runner.bridge.env.command_manager.get_term("reference_trajectory")
    local["sampling"] = command.sampling_state()
    cfg = asdict(runner.cfg)
    bundle = dict(critic=runner.critic.state_dict(), cyclo_lab=local)
    bundle.update(runner.actor_checkpoint())
    if is_supermimic:
        bundle.update(
            format=SUPERMIMIC_FORMAT,
            step=step,
            manifest=supermimic_manifest(runner, task, cfg),
            world_model={"dynamics": runner.dynamics.state_dict(),
                         "sim_context": runner.bridge.sim_context.state_dict()},
            training_stats={"total_steps": runner.metrics.total_steps, "wall_time": runner.metrics.wall_time},
        )
    else:
        for key in ("enc_hidden", "dec_hidden", "wm_hidden"):
            cfg[key] = ",".join(map(str, cfg[key]))
        cfg.update(pi_lr_mult=4.0, wm_mirror=0.5)
        local["decay_start"] = runner.decay_start
        bundle.update(
            args=cfg, iteration=step, wm=runner.dynamics.state_dict(),
            sim_steps=runner.metrics.total_steps, wall_time=runner.metrics.wall_time,
        )
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
    states = {k: [opt.state_dict() for opt in opts] for k, opts in runner.checkpoint_optimizers().items()}
    add_optimizers(bundle, states, runner.family)
    atomic_save(bundle, output / "last.pt")
    return snapshot


def load(runner, bundle, *, training, world_only=False):
    """Restore the runner inputs already decoded by load_checkpoint()."""
    if "dynamics" not in bundle:
        if training or world_only:
            raise ValueError("Policy-only checkpoint cannot resume training or initialize a world model; use last.pt")
        runner.restore_actor(bundle, training=False)
        runner.iteration = bundle["iteration"]
        return
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
    if "critic" in bundle:
        runner.critic.load_state_dict(bundle["critic"])
    elif training:
        raise ValueError("Full training resume requires a critic; use last.pt")
    if training and "optimizers" not in bundle:
        import warnings

        warnings.warn("Model-only checkpoint: optimizer state starts fresh", stacklevel=2)
    if training and "optimizers" in bundle:
        for name, opts in runner.checkpoint_optimizers().items():
            for opt, state in zip(opts, bundle["optimizers"][name], strict=True):
                opt.load_state_dict(state)
    runner.iteration = bundle["iteration"]
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
    if family == "supermimic" and args.world_model_checkpoint and bundle.get("format") == SUPERMIMIC_FORMAT:
        # World-model-only loading does not need actor or manifest metadata.
        bundle = bundle["world_model"]
    bundle = decode(bundle, family)
    if family == "supermimic" and "cfg" not in bundle:
        if not args.world_model_checkpoint:
            raise ValueError("A native world_model.pt only initializes the world model; use --world-model-checkpoint")
    if action == "train" and family == "blinkmimic" and "dynamics" not in bundle:
        raise ValueError("Policy-only checkpoint cannot resume training or initialize a world model; use last.pt")
    return path, bundle


def restore_runner(runner, bundle, *, training, world_only, iterations):
    if bundle:
        load(runner, bundle, training=training, world_only=world_only)
        if training and iterations <= runner.iteration:
            raise ValueError("--iterations must exceed the saved iteration")
