"""Script to benchmark Speed of Simulation and World Model."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import tyro
import wandb
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

import supermimic.tasks  # noqa: F401
from supermimic.representations import extract_actor_goal
from supermimic.runners.run_config import check_representation_matches, load_runner_cfg
from supermimic.runners.supermimic_runner import build_actor, build_world_model
from supermimic.tasks.task_spec import build_actor_obs_td
from supermimic.utils.wandb_artifacts import (
  download_artifact_cached,
  download_model_dir,
)


@dataclass
class BenchmarkConfig:
  wandb_path: str
  task_id: str = "SuperMimic-GenCoord-G1"
  device: str | None = None

  num_envs_list: list[int] = field(
    default_factory=lambda: [64, 128, 256, 512, 1024, 2048, 4096]
  )
  steps: int = 500

  use_policy: bool = True
  inference_mode: bool = False


def benchmark(cfg: BenchmarkConfig) -> None:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  print(f"[INFO] Connecting to WandB run: {cfg.wandb_path}")
  run = wandb.Api().run(cfg.wandb_path)
  runner_cfg = load_runner_cfg(run.config)
  check_representation_matches(cfg.task_id, runner_cfg.representation)

  model_dir = download_model_dir(run)
  bundle = torch.load(model_dir / "world_model.pt", map_location=device)
  policy_state_dict = (
    torch.load(model_dir / "actor.pt", map_location=device) if cfg.use_policy else None
  )

  results = []

  # Resolve the run's linked motion clip once; reused for every num_envs.
  motion_file_path = None
  motion_artifact = next((a for a in run.used_artifacts() if a.type == "motions"), None)
  if motion_artifact:
    motion_file_path = str(download_artifact_cached(motion_artifact) / "motion.npz")

  for n_envs in cfg.num_envs_list:
    print(f"\n{'=' * 40}")
    print(f"Benchmarking num_envs = {n_envs}")
    print(f"{'=' * 40}")

    # ---------------------------
    # 1. Environment / Simulation
    # ---------------------------
    wm = None  # set by stage 1; None means it failed before building.
    try:
      env_cfg = load_env_cfg(cfg.task_id, play=True)
      env_cfg.scene.num_envs = n_envs

      if motion_file_path:
        env_cfg.commands["motion"].motion_file = motion_file_path

      base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
      env = RslRlVecEnvWrapper(base_env)

      wm = build_world_model(runner_cfg, env, device)
      wm.load_bundle(bundle)
      wm.dynamics.eval()
      representation = wm.representation
      sim_ctx = wm.sim_context.read(env.unwrapped)

      policy = None
      if cfg.use_policy:
        assert policy_state_dict is not None  # use_policy gated the load above
        policy = build_actor(runner_cfg, wm, device)
        policy.set_slot_mix({"motion": 1.0})
        policy.load_state_dict(policy_state_dict)
        policy.eval()
      else:
        actions = torch.rand((n_envs, wm.num_actions), device=device) * 2 - 1

      obs, _ = env.reset()

      # Sync CUDA
      if "cuda" in device:
        torch.cuda.synchronize()

      start_time = time.time()
      for _ in range(cfg.steps):
        with torch.inference_mode():
          if policy is not None:
            rep_state = representation.represent(obs["full_state"])
            sensor = representation.compute_actor_sensor(
              rep_state,
              sim_ctx,
              sim_actor_obs=obs["actor"] if runner_cfg.actor_obs == "proprio" else None,
            )
            # Only the shape matters for timing; pad like training does at reset.
            history = sensor.unsqueeze(1).repeat(1, runner_cfg.history_length, 1)
            actor_obs = build_actor_obs_td(
              {"motion": extract_actor_goal(obs["goal"])}, history
            )
            actions = policy(actor_obs)

          obs, _, _, _ = env.step(actions)

      if "cuda" in device:
        torch.cuda.synchronize()
      end_time = time.time()

      dur = end_time - start_time
      sim_sps = (cfg.steps * n_envs) / dur
      label = "Sim + Policy" if cfg.use_policy else "Simulation"
      print(
        f"{label}: {sim_sps:.2f} steps/s (Total time: {dur:.2f}s for {cfg.steps} steps)"
      )

      env.close()
      del env
      del base_env
      if "cuda" in device:
        torch.cuda.empty_cache()

    except Exception as e:
      print(f"Simulation Benchmark Failed for {n_envs}: {e}")
      sim_sps = 0.0

    # ---------------------------
    # 2. World Model Inference
    # ---------------------------
    try:
      # Re-use the world model loaded for the sim benchmark.
      if wm is None:
        continue
      dynamics = wm.dynamics
      representation = wm.representation

      # Create Dummy Inputs
      dummy_rep_state = torch.randn(
        (n_envs, representation.rep_state_dim), device=device
      )
      dummy_action = torch.randn((n_envs, wm.num_actions), device=device)
      dummy_aux = torch.randn((n_envs, 7), device=device)
      dummy_sim_ctx = torch.randn((n_envs, wm.sim_context.output_dim), device=device)

      # Warmup
      for _ in range(10):
        dynamics(dummy_rep_state, dummy_action, dummy_aux, dummy_sim_ctx)

      if "cuda" in device:
        torch.cuda.synchronize()

      start_time = time.time()
      for _ in range(cfg.steps):
        with torch.inference_mode(cfg.inference_mode):
          dynamics(dummy_rep_state, dummy_action, dummy_aux, dummy_sim_ctx)

      if "cuda" in device:
        torch.cuda.synchronize()
      end_time = time.time()

      dur = end_time - start_time
      wm_sps = (cfg.steps * n_envs) / dur
      mode_str = " (Inference Mode)" if cfg.inference_mode else ""
      print(
        f"WorldModel{mode_str}: {wm_sps:.2f} steps/s (Total time: {dur:.2f}s for {cfg.steps} steps)"
      )

      del dynamics, representation, wm
      if "cuda" in device:
        torch.cuda.empty_cache()

    except Exception as e:
      print(f"WorldModel Benchmark Failed for {n_envs}: {e}")
      import traceback

      traceback.print_exc()
      wm_sps = 0.0

    results.append({"num_envs": n_envs, "sim_sps": sim_sps, "wm_sps": wm_sps})

  # Print Summary Table
  print("\n" + "=" * 80)
  print(f"{'Num Envs':<15} | {'Sim SPS':<20} | {'WM SPS':<20} | {'Ratio (WM/Sim)':<20}")
  print("-" * 80)
  for r in results:
    ratio = r["wm_sps"] / r["sim_sps"] if r["sim_sps"] > 0 else 0.0
    print(
      f"{r['num_envs']:<15} | {r['sim_sps']:<20.2f} | {r['wm_sps']:<20.2f} | {ratio:<20.2f}"
    )
  print("=" * 80 + "\n")


def main() -> None:
  args = tyro.cli(BenchmarkConfig)
  benchmark(args)


if __name__ == "__main__":
  main()
