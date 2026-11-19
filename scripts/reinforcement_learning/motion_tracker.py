# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Isaac Lab training, playback and actor export for registered motion trackers."""

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def parse_cli(action, argv):
    from isaaclab.app import add_launcher_args

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-file", nargs="+", type=Path)
    parser.add_argument("--num-envs", "--num_envs", type=int, default=32 if action == "train" else 1)
    parser.add_argument("--iterations", "--max_iterations", type=int, default=1000)
    parser.add_argument("--collection-steps", type=int)
    parser.add_argument("--play-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--hidden", type=int, nargs="+")
    parser.add_argument("--policy-horizon", type=int)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--velocity-fraction", type=float)
    parser.add_argument("--encoder", choices=("motion", "velocity"), help="Trained encoder to play or export")
    parser.add_argument("--velocity-command", nargs=3, type=float, metavar=("VX", "VY", "WZ"))
    parser.add_argument("--checkpoint", type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--resume",
        "--agent.resume",
        nargs="?",
        const="true",
        default="false",
        type=str.lower,
        choices=("true", "false"),
    )
    parser.add_argument("--agent.load-run", "--load_run", dest="run_pattern")
    parser.add_argument("--agent.load-checkpoint", "--load_checkpoint", dest="checkpoint_pattern")
    group.add_argument("--world-model-checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate each BlinkMimic clip from its first frame"
    )
    parser.add_argument("--probe-report", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--headless", action="store_true")
    add_launcher_args(parser)
    if "--physics" not in parser._option_string_actions:
        parser.add_argument("--physics", choices=("newton_mjwarp", "isaacsim_physx"))
    args, overrides = parser.parse_known_args(argv)
    args.resume = args.resume == "true"
    if (args.run_pattern or args.checkpoint_pattern) and (args.checkpoint or args.world_model_checkpoint):
        parser.error("Pass a checkpoint path or run/checkpoint patterns")
    if action == "train" and (args.run_pattern or args.checkpoint_pattern) and not args.resume:
        parser.error("Run/checkpoint selection during training requires --resume")
    if action == "train" and args.encoder:
        parser.error("--encoder selects a trained encoder for play/export; train with --velocity-fraction")
    if action != "train" and (
        args.velocity_fraction is not None or args.resume or args.world_model_checkpoint
    ):
        parser.error("--velocity-fraction, --resume and --world-model-checkpoint apply to training")
    if args.velocity_command and (action != "play" or args.encoder == "motion"):
        parser.error("--velocity-command requires velocity playback")
    return args, overrides


def physics_overrides(requested, overrides, checkpoint):
    if requested:
        return [f"physics={requested}", *overrides]
    if checkpoint and not any(arg.startswith("physics=") for arg in overrides):
        presets = {"PhysxCfg": "isaacsim_physx", "MJWarpSolverCfg": "newton_mjwarp"}
        preset = presets.get(checkpoint.get("physics_backend"))
        if preset:
            return [f"physics={preset}", *overrides]
    return overrides


def select_motions(cfg, task, action, args, bundle):
    cfg.scene.num_envs = args.num_envs
    reference = cfg.commands.reference_trajectory
    defaults = [
        str(Path(p).expanduser().resolve())
        for p in (reference.trajectory_file, *reference.additional_motions)
    ]
    paths = defaults
    if bundle and not args.world_model_checkpoint:
        saved_defaults = [str(ROOT / p) for p in bundle.get("task_motions") or ()]
        if action != "train" or (bundle.get("task") == task and saved_defaults == defaults):
            paths = [ROOT / p for p in bundle["motions"]]
    if args.motion_file:
        paths = args.motion_file
    motions = [str(Path(p).expanduser().resolve(strict=True)) for p in paths]
    reference.trajectory_file, *extra = motions
    reference.additional_motions = tuple(extra)
    reference.start_from_zero = action != "train"
    return motions, defaults


def resolve_learner(settings, args, bundle, experiment):
    for key in ("batch_size", "policy_horizon", "horizon"):
        value = getattr(args, key)
        if value is not None:
            setattr(settings, key, value)
    if args.hidden:
        settings.hidden = settings.world_hidden = tuple(args.hidden)
    if bundle and args.world_model_checkpoint:
        saved = bundle["cfg"]
        settings.world_hidden = tuple(saved.get("world_hidden") or saved["hidden"])

    settings = experiment.configure_learner(settings, args, bundle)
    if (
        min(
            settings.batch_size,
            settings.policy_horizon,
            args.iterations,
            args.num_envs,
            args.collection_steps if args.collection_steps is not None else 1,
        )
        < 1
        or not 2 <= settings.horizon <= settings.buffer_size
    ):
        raise ValueError("Invalid training dimensions or replay horizon")
    return settings


def play(runner, args, encoder, motions):
    viewer = runner.start_playback(encoder, args.velocity_command, motions)
    limit = args.play_steps if viewer or args.play_steps is not None else 1000
    steps = 0
    try:
        while limit is None or steps < limit:
            if viewer and not viewer.window.visible:
                break
            if viewer:
                viewer.before_step()
            runner.play_step()
            if viewer:
                viewer.draw()
            steps += 1
    finally:
        if viewer:
            viewer.close()
    print(f"[INFO] Played {steps} Isaac Lab steps with the {encoder} encoder", flush=True)


def selected_encoder(runner, args):
    encoder = args.encoder or ("velocity" if args.velocity_command else None)
    if encoder and encoder not in runner.active_encoders:
        raise ValueError(f"Encoder {encoder!r} was not trained; available: {', '.join(runner.active_encoders)}")
    return encoder


def run_export(runner, args, path):
    from cyclo_lab.manager_based.mimic.export import export_actors

    encoder = selected_encoder(runner, args)
    encoders = [encoder] if encoder else runner.active_encoders
    export_actors(runner, args.output or path.parent / "exported", encoders=encoders)


def run_playback(runner, args, path, motions, experiment):
    from cyclo_lab.manager_based.mimic.export import export_actors

    encoder = selected_encoder(runner, args)
    if not args.probe_report:
        export_actors(runner, args.output or path.parent / "exported", encoders=runner.active_encoders)
    if args.probe_report or args.evaluate:
        experiment.evaluate(runner, path, args.probe_report)
        return
    play(runner, args, encoder or runner.active_encoders[0], motions)


def main(action, task, argv, *, family):
    from importlib import import_module

    os.environ.setdefault("TORCHINDUCTOR_WORKER_START", "spawn")
    import gymnasium as gym
    import torch
    from isaaclab.app import launch_simulation
    from isaaclab_tasks.utils import resolve_task_config
    from cyclo_lab.manager_based.mimic import checkpoint
    from cyclo_lab.manager_based.mimic.training_loop import run_training_loop
    from cyclo_lab.manager_based.mimic.tracking_contract import validate_native_cfg

    args, overrides = parse_cli(action, argv)
    experiment = import_module(f"cyclo_lab.manager_based.mimic.{family}.core.experiment")
    if (args.evaluate or args.probe_report) and (experiment.evaluate is None or action != "play"):
        raise ValueError("Clip evaluation requires BlinkMimic play")
    root = ROOT / "logs" / family
    path, bundle = checkpoint.load_checkpoint(action, args, family, task, root)
    if args.eager:
        torch._dynamo.config.disable = True
    overrides = physics_overrides(args.physics, overrides, None if args.world_model_checkpoint else bundle)
    args.physics = None
    cfg, settings = resolve_task_config(task, "learner_cfg_entry_point", play_mode=False, overrides=overrides)
    motions, task_motions = select_motions(cfg, task, action, args, bundle)
    settings = resolve_learner(settings, args, bundle, experiment)
    experiment.configure_environment(cfg, action, args)
    validate_native_cfg(cfg)

    args.require_kit = True
    with launch_simulation(cfg, args):
        env = gym.make(task, cfg=cfg)
        try:
            env.reset()
            runner = experiment.Runner(env, settings)
            solver = getattr(cfg.sim.physics, "solver_cfg", None)
            backend = type(solver if solver is not None else cfg.sim.physics).__name__
            contract = checkpoint.restore_runner(
                runner, cfg, bundle, backend, training=action == "train",
                world_only=bool(args.world_model_checkpoint), iterations=args.iterations,
            )
            print(
                f"[INFO] {task}: {backend}, {args.num_envs} envs, {len(motions)} clips, dt={runner.bridge.step_dt}",
                flush=True,
            )
            if action == "train":
                run_training_loop(
                    runner, args, root, task, contract, motions, backend, task_motions, experiment.training_probe
                )
            elif action == "play":
                run_playback(runner, args, path, motions, experiment)
            else:
                run_export(runner, args, path)
        finally:
            env.close()
