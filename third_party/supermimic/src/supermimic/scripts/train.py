"""Script to train World Model using SuperMimicRunner."""

from __future__ import annotations

import os
from dataclasses import dataclass

import mjlab.tasks  # noqa: F401
import tyro
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg

import supermimic.tasks  # noqa: F401
from supermimic.runners.supermimic_runner import SuperMimicRunner, SuperMimicRunnerCfg
from supermimic.scripts.train_utils import (
  create_env,
  download_registry_motions,
  setup_device,
)


@dataclass
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  runner: SuperMimicRunnerCfg

  # Experiment & Environment
  num_envs: int = 4096
  device: str | None = None
  registry_name: str | None = "supermimic/wandb-registry-motions/lafan_run1_subject5"
  motion_file: str | None = None

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    env_cfg = load_env_cfg(task_id, play=False)
    return TrainConfig(
      env=env_cfg,
      runner=SuperMimicRunnerCfg(),
    )


def main() -> None:
  # Parse first argument to choose the task.
  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    config=mjlab.TYRO_FLAGS,
  )

  device = setup_device(args.device)
  print(f"[INFO] Loading environment: {chosen_task}")

  if args.motion_file:
    # Local motions handed straight to MotionLoader (globs/concatenates).
    if "," in args.motion_file:
      args.env.commands["motion"].motion_file = [
        os.path.expanduser(p.strip()) for p in args.motion_file.split(",")
      ]
    else:
      args.env.commands["motion"].motion_file = os.path.expanduser(args.motion_file)
    args.registry_name = None  # avoid linking a stale registry artifact for lineage
    print(f"[INFO] Using local motion file: {args.motion_file}")
  else:
    motion_file = download_registry_motions(args.registry_name)
    if motion_file is not None:
      args.env.commands["motion"].motion_file = motion_file

  # Auto-detect representation from task_id if not explicitly set
  if "MaxCoord" in chosen_task and args.runner.representation == "gencoord":
    args.runner.representation = "maxcoord"
    print("[INFO] Auto-detected maxcoord representation from task_id")

  args.env.commands["motion"].sampling_mode = (
    "external" if args.runner.sampling_mode == "return" else args.runner.sampling_mode
  )

  env = create_env(args.env, args.num_envs, device)

  # Init Runner
  runner = SuperMimicRunner(
    env=env,
    cfg=args.runner,
    task_id=chosen_task,
    registry_name=args.registry_name,
    device=device,
  )

  runner.run()


if __name__ == "__main__":
  main()
