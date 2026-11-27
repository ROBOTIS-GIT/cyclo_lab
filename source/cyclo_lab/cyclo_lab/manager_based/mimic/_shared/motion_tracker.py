# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
#
# Author: Kiwoong Park
"""Isaac Lab training, playback and actor export for registered motion trackers."""

import argparse
from copy import deepcopy
from dataclasses import fields, is_dataclass
import os
from pathlib import Path

from .. import REPO_ROOT


def parse_cli(action, argv):
    from isaaclab.app import add_launcher_args

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-file", nargs="+", type=Path)
    parser.add_argument("--num-envs", "--num_envs", type=int)
    parser.add_argument("--iterations", "--max_iterations", type=int, default=1000)
    parser.add_argument("--collection-steps", type=int)
    parser.add_argument("--play-steps", type=int)
    parser.add_argument("--encoder", choices=("motion", "velocity"), help="Trained encoder to play")
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
    parser.add_argument("--agent.load-run", "--agent.load_run", "--load_run", dest="run_pattern")
    parser.add_argument("--agent.load-checkpoint", "--agent.load_checkpoint", "--load_checkpoint", dest="checkpoint_pattern")
    group.add_argument("--world-model-checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate each BlinkMimic clip from its first frame"
    )
    parser.add_argument("--probe-report", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--eager", action="store_true", default=None)
    parser.add_argument("--headless", action="store_true")
    add_launcher_args(parser)
    if "--physics" not in parser._option_string_actions:
        parser.add_argument("--physics", choices=("newton_mjwarp", "isaacsim_physx"))
    # Keep key=value overrides out of argparse's variable-length motion list.
    options, overrides = [], []
    for value in argv:
        target = overrides if "=" in value and not value.startswith("-") else options
        target.append(value)
    args, unknown = parser.parse_known_args(options)
    overrides.extend(unknown)
    if any(value.startswith("--") for value in overrides):
        parser.error("Unknown option; use agent.<setting>=<value> for learner settings")
    args.resume = args.resume == "true"
    if (args.run_pattern or args.checkpoint_pattern) and (args.checkpoint or args.world_model_checkpoint):
        parser.error("Pass a checkpoint path or run/checkpoint patterns")
    if action == "train" and (args.run_pattern or args.checkpoint_pattern) and not args.resume:
        parser.error("Run/checkpoint selection during training requires --resume")
    if action == "train" and args.encoder:
        parser.error("--encoder selects a trained encoder for play; train with agent.slot_mix")
    if action != "train" and (
        args.resume or args.world_model_checkpoint
    ):
        parser.error("--resume and --world-model-checkpoint apply to training")
    if args.velocity_command and (action != "play" or args.encoder == "motion"):
        parser.error("--velocity-command requires velocity playback")
    return args, overrides


def select_motions(cfg, action, args):
    reference = cfg.commands.reference_trajectory
    defaults = [
        str(Path(p).expanduser().resolve())
        for p in (reference.trajectory_file, *reference.additional_motions)
    ]
    paths = defaults
    if args.motion_file:
        paths = args.motion_file
    motions = [str(Path(p).expanduser().resolve(strict=True)) for p in paths]
    reference.trajectory_file, *extra = motions
    reference.additional_motions = tuple(extra)
    reference.start_from_zero = action != "train"
    return motions, defaults


def config_parent(config, path):
    """Resolve a declared learner field, including nested dataclasses and dictionaries."""
    names = path.split(".")
    for index, name in enumerate(names):
        if is_dataclass(config) and name not in {field.name for field in fields(config)}:
            raise ValueError(f"Unknown learner option: agent.{path}")
        if isinstance(config, dict) and name not in config:
            raise ValueError(f"Unknown learner option: agent.{path}")
        if index == len(names) - 1:
            return config, name
        config = config[name] if isinstance(config, dict) else getattr(config, name)


def resolve_learner(settings, args, bundle, experiment, *, overrides=(), training=True):
    explicit = [value.partition("=")[0][6:] for value in overrides if value.startswith("agent.") and "=" in value]
    for path in explicit:
        config_parent(settings, path)
    if bundle and not args.world_model_checkpoint:
        if hasattr(settings, "representation") and settings.representation != bundle["cfg"]["representation"]:
            raise ValueError("Checkpoint representation differs from the selected task")
        saved = experiment.restore_config(bundle["cfg"])
        if training and hasattr(settings, "wm_structured"):
            saved.wm_structured = settings.wm_structured
        # resolve_task_config already parsed these values. Keep explicit agent
        # overrides, including values equal to the current task's defaults.
        for path in explicit:
            source, name = config_parent(settings, path)
            target, _ = config_parent(saved, path)
            value = deepcopy(source[name] if isinstance(source, dict) else getattr(source, name))
            if isinstance(target, dict):
                target[name] = value
            else:
                setattr(target, name, value)
        settings = saved
    settings = experiment.configure_learner(settings, args, bundle)
    if args.iterations < 1 or (args.collection_steps is not None and args.collection_steps < 1):
        raise ValueError("Iteration and collection counts must be positive")
    return settings


def play(runner, args, encoder, motions):
    viewer = runner.start_playback(encoder, args.velocity_command, motions)
    limit = args.play_steps if viewer or args.play_steps is not None else 1_000_000
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


def run_playback(runner, args, path, motions, experiment):
    from cyclo_lab.manager_based.mimic._shared.export import export_actors

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
    from cyclo_lab.manager_based.mimic._shared import checkpoint
    from cyclo_lab.manager_based.mimic._shared.training_loop import run_training_loop
    from cyclo_lab.manager_based.mimic._shared.tracking_contract import validate_native_cfg

    args, overrides = parse_cli(action, argv)
    experiment = import_module(f"cyclo_lab.manager_based.mimic.{family}.experiment")
    if (args.evaluate or args.probe_report) and (experiment.evaluate is None or action != "play"):
        raise ValueError("Clip evaluation requires BlinkMimic play")
    root = REPO_ROOT / "logs" / family
    path, bundle = checkpoint.load_checkpoint(action, args, family, task, root)
    if args.eager:
        torch._dynamo.config.disable = True
    if args.physics:
        overrides = [f"physics={args.physics}", *overrides]
    args.physics = None
    cfg, settings = resolve_task_config(task, "learner_cfg_entry_point", play_mode=action != "train", overrides=overrides)
    if args.num_envs is not None:
        cfg.scene.num_envs = args.num_envs
    if cfg.scene.num_envs < 1:
        raise ValueError("Number of environments must be positive")
    motions, task_motions = select_motions(cfg, action, args)
    settings = resolve_learner(
        settings, args, bundle, experiment, overrides=overrides, training=action == "train"
    )
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
            checkpoint.restore_runner(
                runner, bundle, training=action == "train",
                world_only=bool(args.world_model_checkpoint), iterations=args.iterations,
            )
            print(
                f"[INFO] {task}: {backend}, {cfg.scene.num_envs} envs, {len(motions)} clips, dt={runner.bridge.step_dt}",
                flush=True,
            )
            if action == "train":
                run_training_loop(
                    runner, args, root, task, motions, backend, task_motions, experiment.training_probe
                )
            else:
                run_playback(runner, args, path, motions, experiment)
        finally:
            env.close()
