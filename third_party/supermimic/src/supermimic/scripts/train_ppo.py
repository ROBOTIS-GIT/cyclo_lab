"""Script to test PPORunner."""

from __future__ import annotations

import os
from dataclasses import dataclass

import mjlab.tasks  # noqa: F401
import tyro
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg

import supermimic.tasks  # noqa: F401
from supermimic.runners.ppo_runner import PpoRunner, PpoRunnerCfg
from supermimic.scripts.train_utils import (
  create_env,
  download_registry_motions,
  setup_device,
)


@dataclass
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  runner: PpoRunnerCfg

  # Experiment & Environment
  num_envs: int = 1024
  device: str | None = None
  registry_name: str | None = "supermimic/wandb-registry-motions/lafan_run1_subject5"
  motion_file: str | None = None

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    env_cfg = load_env_cfg(task_id, play=False)
    return TrainConfig(
      env=env_cfg,
      runner=PpoRunnerCfg(),
    )


def main() -> None:
  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=(tyro.conf.AvoidSubcommands, tyro.conf.FlagConversionOff),
  )
  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    config=(tyro.conf.AvoidSubcommands, tyro.conf.FlagConversionOff),
  )

  device = setup_device(args.device)
  print(f"[INFO] Loading environment: {chosen_task}")

  if args.motion_file:
    if "," in args.motion_file:
      args.env.commands["motion"].motion_file = [
        os.path.expanduser(p.strip()) for p in args.motion_file.split(",")
      ]
    else:
      args.env.commands["motion"].motion_file = os.path.expanduser(args.motion_file)
    args.registry_name = None
    print(f"[INFO] Using local motion file: {args.motion_file}")
  else:
    motion_file = download_registry_motions(args.registry_name)
    if motion_file is not None:
      args.env.commands["motion"].motion_file = motion_file

  if "MaxCoord" in chosen_task and args.runner.representation == "gencoord":
    args.runner.representation = "maxcoord"
    print("[INFO] Auto-detected maxcoord representation from task_id")

  env = create_env(args.env, args.num_envs, device)

  # Init Runner
  runner = PpoRunner(
    env=env,
    cfg=args.runner,
    task_id=chosen_task,
    registry_name=args.registry_name,
    device=device,
  )

  runner.run()


if __name__ == "__main__":
  main()
