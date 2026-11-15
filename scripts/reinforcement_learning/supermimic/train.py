# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Collect Isaac Lab states, fit SuperMimic dynamics, and optimize imagined rollouts."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys

# Kit/CUDA are not fork-safe. Configure only this learner process before Torch imports.
os.environ.setdefault("TORCHINDUCTOR_WORKER_START", "spawn")
if "--eager" in sys.argv:
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "source/cyclo_lab"))


def main():
    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab_tasks.utils import resolve_task_config
    import cyclo_lab  # noqa: F401

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task")
    parser.add_argument("--motion-file", type=Path)
    parser.add_argument("--num-envs", "--num_envs", type=int, default=32)
    parser.add_argument("--iterations", "--max_iterations", type=int, default=1000)
    parser.add_argument("--collection-steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--policy-horizon", type=int, default=16)
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--latent", type=int, default=32)
    parser.add_argument(
        "--eager",
        action="store_true",
        help="Same updates without compile or CUDA graphs",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--play-steps", type=int, default=1000)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Check live FK, updates and checkpoint reload",
    )
    add_launcher_args(parser)
    args, overrides = parser.parse_known_args()
    if args.eager:
        import torch

        torch._dynamo.config.disable = True
    if args.num_envs < 1 or args.iterations < 1 or args.collection_steps < 2:
        parser.error("Require positive envs/iterations and >=2 collection steps")
    if args.play and not args.checkpoint:
        parser.error("--play requires --checkpoint")
    saved = None
    if args.checkpoint:
        import torch

        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if saved.get("format") != "cyclo-supermimic-isaac-v1":
            raise ValueError("Expected a native Isaac SuperMimic checkpoint")
        args.task = args.task or saved["task"]
        if args.motion_file is None:
            args.motion_file = Path(saved["motion_file"])
            if not args.motion_file.is_file():
                parser.error(
                    "Saved motion path is unavailable; provide --motion-file for this checkout"
                )
    args.task = args.task or "Cyclo-SuperMimic-K1-Rev1-Dance1"
    if args.task not in (
        "Cyclo-SuperMimic-K1-Rev1-Dance1",
        "Cyclo-SuperMimic-K1-Rev1-Dance2",
    ):
        parser.error(
            "SuperMimic currently supports K1 Dance1/Dance2 configurations; use --motion-file for other clips"
        )
    cfg, _ = resolve_task_config(
        args.task,
        "supermimic_cfg_entry_point",
        play_mode=args.play,
        overrides=["physics=isaacsim_physx", *overrides],
    )
    cfg.scene.num_envs = args.num_envs
    if args.motion_file:
        cfg.commands.reference_trajectory.trajectory_file = str(
            args.motion_file.resolve()
        )
    from cyclo_lab.learning.supermimic.motion import validate_motion

    print(
        "MOTION_CONTRACT",
        json.dumps(validate_motion(cfg.commands.reference_trajectory.trajectory_file)),
        flush=True,
    )
    output = args.output or ROOT / "logs/supermimic" / datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )
    with launch_simulation(cfg, args):
        import gymnasium as gym
        import torch
        from cyclo_lab.learning.supermimic.runner import (
            IsaacSuperMimicRunner,
            RunnerCfg,
        )

        if args.eager:
            torch._dynamo.config.disable = True
        env = gym.make(args.task, cfg=cfg)
        try:
            env.reset()
            runner_cfg = RunnerCfg(
                hidden=tuple(args.hidden),
                latent=args.latent,
                batch_size=args.batch_size,
                horizon=args.horizon,
                policy_horizon=args.policy_horizon,
                collection_steps=args.collection_steps,
                eager=args.eager,
            )
            if saved is not None:
                runner_cfg = RunnerCfg(**saved["cfg"])
                runner_cfg.eager = args.eager
            runner = IsaacSuperMimicRunner(env, runner_cfg)
            if args.checkpoint:
                runner.load(args.checkpoint)
            print(
                "ISAAC_BACKEND",
                type(env.unwrapped).__module__,
                type(cfg.sim.physics).__name__,
                flush=True,
            )
            if args.verify:
                print("ISAAC_FK_PARITY", runner.bridge.check_fk(), flush=True)
            if args.play:
                with torch.no_grad():
                    runner.actor.eval()
                    for _ in range(args.play_steps):
                        action = runner.actor(
                            runner.actor_obs(
                                runner.bridge.state(), runner.bridge.frame_ids()
                            )
                        ).clamp(-10, 10)
                        env.step(action)
                        if not torch.isfinite(runner.bridge.state()).all():
                            raise RuntimeError("Nonfinite Isaac state during playback")
                print("ISAAC_PLAY_PASS", args.play_steps, flush=True)
                return
            output.mkdir(parents=True, exist_ok=True)
            for _ in range(args.iterations):
                runner.collect(max(args.collection_steps, args.horizon))
                if args.verify:
                    print(
                        "ISAAC_FK_PARITY_AFTER_COLLECTION",
                        runner.bridge.check_fk(atol=5e-3),
                        flush=True,
                    )
                metrics = runner.update()
                print(
                    json.dumps(
                        {
                            "iteration": runner.iteration,
                            "isaac_steps": runner.collected_steps,
                            **metrics,
                        }
                    ),
                    flush=True,
                )
                checkpoint = output / f"model_{runner.iteration}.pt"
                runner.save(checkpoint)
                if args.verify:
                    before = {
                        k: v.clone() for k, v in runner.actor.state_dict().items()
                    }
                    runner.load(checkpoint)
                    assert all(
                        torch.equal(v, runner.actor.state_dict()[k])
                        for k, v in before.items()
                    )
                    print("ISAAC_CHECKPOINT_RELOAD_PASS", str(checkpoint), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
