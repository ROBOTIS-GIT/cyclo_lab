"""Select, save, validate and restore native motion tracker checkpoints."""

from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile
import torch

from . import REPO_ROOT


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


def save(runner, output, contract, motions, physics_backend=None, *, task=None, task_motions=None):
    step = runner.iteration
    bundle = dict(
        format="cyclo_lab.mimic.v1",
        task=task,
        task_motions=checkpoint_motion_paths(task_motions) if task_motions is not None else None,
        cfg=asdict(runner.cfg),
        contract=contract,
        physics_backend=physics_backend,
        iteration=step,
        motions=checkpoint_motion_paths(motions),
        joint_names=list(runner.bridge.robot.joint_names),
    )
    if hasattr(runner, "dynamics_contract"):
        bundle["dynamics_contract"] = runner.dynamics_contract
        command = runner.bridge.env.command_manager.get_term("reference_trajectory")
        bundle["sampling"] = command.sampling_state()
    bundle.update(runner.actor_checkpoint())
    bundle["critic"] = runner.critic.state_dict()
    bundle["dynamics"] = runner.dynamics.state_dict()
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
    if bundle["joint_names"] != list(runner.bridge.robot.joint_names):
        raise ValueError("Checkpoint articulation order differs from the current training task")
    try:
        runner.dynamics.load_state_dict(bundle["dynamics"])
    except RuntimeError as exc:
        raise ValueError(f"World-model weights are incompatible with this task: {exc}") from exc
    if world_only:
        return
    if training and "sampling" in bundle:
        runner.bridge.env.command_manager.get_term("reference_trajectory").restore_sampling(
            bundle["sampling"]
        )
    runner.restore_actor(bundle)
    runner.critic.load_state_dict(bundle["critic"])
    if training and "optimizers" in bundle:
        for name, opts in runner.checkpoint_optimizers().items():
            for opt, state in zip(opts, bundle["optimizers"][name], strict=True):
                opt.load_state_dict(state)
    runner.iteration = bundle["iteration"]


def find_checkpoint(root, task, path, run_pattern=None, checkpoint_pattern=None):
    if path is not None:
        path = Path(path).expanduser()
        return path / "last.pt" if path.is_dir() else path
    if run_pattern or checkpoint_pattern:
        import re

        directory = root / task
        runs = sorted(
            p for p in directory.iterdir() if p.is_dir() and re.fullmatch(run_pattern or ".*", p.name)
        )
        if not runs:
            raise FileNotFoundError(f"No run matching {run_pattern!r} under {directory}")
        files = [p for p in runs[-1].glob("*.pt") if re.fullmatch(checkpoint_pattern or "last[.]pt", p.name)]
        if not files:
            raise FileNotFoundError(f"No matching checkpoint under {runs[-1]}")
        return max(files, key=lambda p: (p.stem == "last", int(p.stem[6:]) if p.stem[6:].isdecimal() else -1))
    runs = sorted((root / task).glob("*/last.pt"))
    if not runs:
        raise FileNotFoundError(f"No last.pt under {root / task}")
    return runs[-1]


def load_checkpoint(action, args, family, task, root):
    if args.world_model_checkpoint and args.checkpoint:
        raise ValueError("Choose a full checkpoint or a world-model checkpoint")
    if action == "train" and not (args.checkpoint or args.resume or args.world_model_checkpoint):
        return None, None
    path = find_checkpoint(
        root,
        task,
        args.checkpoint or args.world_model_checkpoint,
        args.run_pattern,
        args.checkpoint_pattern,
    )
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if bundle.get("format") != "cyclo_lab.mimic.v1" or bundle["contract"]["family"] != family:
        raise ValueError("Checkpoint algorithm or format does not match this Isaac Lab task")
    return path, bundle


def restore_runner(runner, cfg, bundle, backend, *, training, world_only, iterations):
    from cyclo_lab.manager_based.mimic.tracking_contract import check_dynamics, dynamics_contract

    bridge = runner.bridge
    if training:
        runner.dynamics_contract = dynamics_contract(cfg)
        if bundle and not world_only:
            check_dynamics(bundle, runner.dynamics_contract)
    contract = dict(
        family=runner.family, step_dt=bridge.step_dt, action=[v.tolist() for v in bridge.action_transform()]
    )
    if (
        training
        and bundle
        and bundle.get("physics_backend", backend) != backend
        and not world_only
    ):
        raise ValueError("Full training resume requires the original physics backend")
    if bundle:
        if contract != bundle["contract"]:
            raise ValueError("Checkpoint timing or actuator settings differ")
        load(runner, bundle, training=training, world_only=world_only)
        if training and iterations <= runner.iteration:
            raise ValueError("--iterations must exceed the saved iteration")
    return contract
