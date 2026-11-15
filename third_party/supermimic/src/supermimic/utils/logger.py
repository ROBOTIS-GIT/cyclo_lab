from __future__ import annotations

import statistics
import time
from collections import deque

import torch
import wandb


class Logger:
  """Episode statistics of the real env, logged as one W&B row and one console
  block per log call."""

  def __init__(
    self,
    run_name: str | None,
    num_envs: int,
    max_episode_length: int,
    device: str,
  ) -> None:
    self.run_name = run_name
    self.num_envs = num_envs
    self.max_episode_length = max_episode_length
    self.device = device

    self.tot_timesteps = 0
    self.tot_time = 0.0

    # Create buffers
    self.ep_extras = []
    self.rewbuffer = deque(maxlen=100)
    self.lenbuffer = deque(maxlen=100)
    self.cur_reward_sum = torch.zeros(
      self.num_envs, dtype=torch.float, device=self.device
    )
    self.cur_episode_length = torch.zeros(
      self.num_envs, dtype=torch.float, device=self.device
    )

  def process_env_step(
    self,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    extras: dict,
  ) -> None:
    """Add metrics from the environment step to the buffers."""
    if "episode" in extras:
      self.ep_extras.append(extras["episode"])
    elif "log" in extras:
      self.ep_extras.append(extras["log"])

    self.cur_reward_sum += rewards
    self.cur_episode_length += 1

    flush_mask = terminated | (self.cur_episode_length >= self.max_episode_length)
    new_ids = flush_mask.nonzero(as_tuple=False)
    if len(new_ids) > 0:
      self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
      self.lenbuffer.extend(
        self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
      )
      self.cur_reward_sum[new_ids] = 0
      self.cur_episode_length[new_ids] = 0

  def _episode_extras(self) -> dict[str, float]:
    """Mean of each episode extra over the episodes since the last log."""
    means = {}
    all_keys = dict.fromkeys(k for extra in self.ep_extras for k in extra)
    for key in all_keys:
      infotensor = torch.tensor([], device=self.device)
      for ep_info in self.ep_extras:
        if key not in ep_info:
          continue
        val = ep_info[key]
        if not isinstance(val, torch.Tensor):
          val = torch.tensor([val], device=self.device)
        else:
          val = val.to(self.device).reshape(-1)

        if val.ndim == 0:
          val = val.unsqueeze(0)
        infotensor = torch.cat((infotensor, val))

      if len(infotensor) > 0:
        means[key] = torch.mean(infotensor).item()
    return means

  def _build_console_log(
    self,
    it: int,
    total_it: int,
    fps: int,
    collect_time: float,
    wm_learn_time: float,
    policy_learn_time: float,
    losses: dict[str, float],
    extras: dict[str, float],
    iteration_time: float,
    width: int,
    pad: int,
  ) -> str:
    """Build the formatted console log string."""
    log_string = f"""{"#" * width}\n"""
    log_string += (
      f"""\033[1m{f" Learning iteration {it}/{total_it} ".center(width)}\033[0m \n\n"""
    )

    log_string += f"""{"Run name:":>{pad}} {self.run_name}\n"""

    log_string += (
      f"""{"Total steps:":>{pad}} {self.tot_timesteps} \n"""
      f"""{"Steps per second:":>{pad}} {fps:.0f} \n"""
      f"""{"Collection time:":>{pad}} {collect_time:.3f}s \n"""
      f"""{"WM Learning time:":>{pad}} {wm_learn_time:.3f}s \n"""
      f"""{"Policy Learning time:":>{pad}} {policy_learn_time:.3f}s \n"""
    )

    for key, value in losses.items():
      log_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""

    if len(self.rewbuffer) > 0:
      log_string += (
        f"""{"Mean reward:":>{pad}} {statistics.mean(self.rewbuffer):.2f}\n"""
      )
      log_string += (
        f"""{"Mean episode length:":>{pad}} {statistics.mean(self.lenbuffer):.2f}\n"""
      )

    for key, value in extras.items():
      label = f"{key}:" if "/" in key else f"Episode {key}:"
      log_string += f"""{label:>{pad}} {value:.4f}\n"""

    done_it = it + 1
    remaining_it = total_it - done_it
    eta = self.tot_time / done_it * remaining_it if done_it > 0 else 0.0
    log_string += (
      f"""{"-" * width}\n"""
      f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s\n"""
      f"""{"Time elapsed:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
      f"""{"ETA:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(eta))}\n"""
    )

    return log_string

  def log(
    self,
    it: int,
    total_it: int,
    collect_time: float,
    wm_learn_time: float,
    policy_learn_time: float,
    loss_dict: dict[str, float | torch.Tensor],
    num_collected_steps: int,
    width: int = 80,
    pad: int = 40,
  ) -> None:
    iteration_time = collect_time + wm_learn_time + policy_learn_time
    self.tot_timesteps += num_collected_steps
    self.tot_time += iteration_time
    fps = int(num_collected_steps / (iteration_time + 1e-6))

    extras = self._episode_extras()
    # Device stats reach the host in one transfer.
    on_device = [k for k, v in loss_dict.items() if isinstance(v, torch.Tensor)]
    values = (
      torch.stack([loss_dict[k] for k in on_device]).tolist() if on_device else []
    )
    losses = {**loss_dict, **dict(zip(on_device, values, strict=True))}
    row = losses | {
      "Perf/total_fps": fps,
      "Perf/collection_time": collect_time,
      "Perf/wm_learning_time": wm_learn_time,
      "Perf/policy_learning_time": policy_learn_time,
    }
    if len(self.rewbuffer) > 0:
      row["Train/mean_reward"] = statistics.mean(self.rewbuffer)
      row["Train/mean_episode_length"] = statistics.mean(self.lenbuffer)
    row |= {k if "/" in k else f"Episode/{k}": v for k, v in extras.items()}
    wandb.log(row, step=it)
    print(
      self._build_console_log(
        it,
        total_it,
        fps,
        collect_time,
        wm_learn_time,
        policy_learn_time,
        losses,
        extras,
        iteration_time,
        width,
        pad,
      )
    )
    self.ep_extras.clear()
