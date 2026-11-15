from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from supermimic.representations import extract_actor_goal
from supermimic.runners.supermimic_runner import BaseRunner, BaseRunnerCfg
from supermimic.utils.wandb_artifacts import log_model_artifact


@dataclass
class PpoRunnerCfg(BaseRunnerCfg):
  max_iterations: int = 25000
  log_interval: int = 1
  save_interval: int = 10
  # Real-env collection steps per outer iteration.
  num_collect_steps: int = 10
  # PPO Parameters
  num_steps_per_env: int = 32
  num_learning_epochs: int = 5
  num_mini_batches: int = 4
  clip_param: float = 0.2
  gamma: float = 0.99
  lam: float = 0.95
  desired_kl: float = 0.01
  max_grad_norm: float = 1.0
  learning_rate: float = 1e-3
  value_loss_coef: float = 1.0
  entropy_coef: float = 0.005
  batch_size: int = 4096
  # Policy Architecture
  actor_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
  critic_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
  activation: str = "elu"
  init_noise_std: float = 1.0


class PpoRunner(BaseRunner[PpoRunnerCfg]):
  def _init_policy(self) -> None:
    num_actions = self._num_actions

    actor_input_dim, critic_input_dim = self._policy_input_dims()

    print(
      f"[INFO] Actor Input Dim: {actor_input_dim}, Critic Input Dim: {critic_input_dim}"
    )

    actor_model = self._build_mlp(
      "actor",
      actor_input_dim,
      num_actions,
      hidden_dims=self.cfg.actor_hidden_dims,
      activation=self.cfg.activation,
      distribution_cfg={
        "class_name": "rsl_rl.modules.distribution.GaussianDistribution",
        "init_std": self.cfg.init_noise_std,
      },
    )
    critic_model = self._build_mlp(
      "critic",
      critic_input_dim,
      1,
      hidden_dims=self.cfg.critic_hidden_dims,
      activation=self.cfg.activation,
    )

    obs_template = TensorDict(
      {
        "actor": torch.zeros(self.cfg.batch_size, actor_input_dim, device=self.device),
        "critic": torch.zeros(
          self.cfg.batch_size, critic_input_dim, device=self.device
        ),
      },
      batch_size=[self.cfg.batch_size],
      device=self.device,
    )
    storage = RolloutStorage(
      "rl",
      self.cfg.batch_size,
      self.cfg.num_steps_per_env,
      obs_template,
      [num_actions],
      device=self.device,
    )

    self.alg = PPO(
      actor=actor_model,
      critic=critic_model,
      storage=storage,
      device=self.device,
      num_learning_epochs=self.cfg.num_learning_epochs,
      num_mini_batches=self.cfg.num_mini_batches,
      clip_param=self.cfg.clip_param,
      gamma=self.cfg.gamma,
      lam=self.cfg.lam,
      value_loss_coef=self.cfg.value_loss_coef,
      entropy_coef=self.cfg.entropy_coef,
      learning_rate=self.cfg.learning_rate,
      max_grad_norm=self.cfg.max_grad_norm,
      desired_kl=self.cfg.desired_kl,
    )

    self.wm_env = self._create_wm_env(self.cfg.batch_size)

  def imagine(self) -> dict[str, float]:
    """Roll out the WM with the current policy and run one PPO update on the imagined batch."""
    self.alg.eval_mode()
    self.alg.storage.clear()

    with torch.no_grad():
      obs = self.wm_env.reset()
      for _ in range(self.cfg.num_steps_per_env):
        actions = self.alg.act(obs)
        obs, rewards, dones, _ = self.wm_env.step(actions)
        self.alg.process_env_step(obs, rewards, dones, {})
      self.alg.compute_returns(obs)

    return self.alg.update()

  def collect_data(self) -> int:
    """Collect real-env transitions into the replay buffer."""
    obs = self.env.get_observations()

    with torch.inference_mode():
      for _ in range(self.cfg.num_collect_steps):
        curr_states = obs["full_state"].to(self.device)
        curr_goals = obs["goal"].to(self.device)
        sim_actor_obs = obs["actor"].to(self.device)

        actor_goal = extract_actor_goal(curr_goals)
        rep_state = self._representation.represent(curr_states)
        aux_state = self._representation.extract_aux_state(curr_states)
        sim_ctx = self._sim_context.read(self.env.unwrapped)
        sensor = self._representation.compute_actor_sensor(
          rep_state,
          sim_ctx,
          aux_state=aux_state,
          sim_actor_obs=sim_actor_obs,
          noisy=True,
        )
        actor_input = self._representation.build_actor_obs(
          self._collection_history(sensor), actor_goal
        )
        actor_td = self._actor_td(actor_input)
        actions = self.alg.actor(actor_td, stochastic_output=True)

        extra_fields = {"sim_ctx": sim_ctx}
        if self._stores_actor_obs:
          extra_fields["actor_obs"] = sim_actor_obs
        next_obs, rewards, dones, extras = self._step_and_store(
          actions,
          curr_states,
          actor_td=actor_td,
          extra_fields=extra_fields,
        )
        self._advance_sim_history(next_obs["actor"].to(self.device), dones)
        terminated = self.env.unwrapped.termination_manager.terminated
        self.logger.process_env_step(rewards, terminated, extras)
        obs = next_obs

    self._flush_normalization()
    return self.cfg.num_collect_steps * self.env.num_envs

  def _actor_td(self, actor_obs: torch.Tensor) -> TensorDict:
    return TensorDict(
      {"actor": actor_obs}, batch_size=actor_obs.shape[:-1], device=self.device
    )

  def _update_actor_normalization(self, actor_inputs: tuple[TensorDict, ...]) -> None:
    # EmpiricalNormalization.update is a no-op in eval mode.
    self.alg.actor.train()
    for actor_td in actor_inputs:
      self.alg.actor.update_normalization(actor_td)

  def run(self) -> None:
    log_dir = self._create_log_dir("ppo")
    self._init_wandb(log_dir, name_suffix="-ppo")
    self._seed_sim_history()

    print(f"[INFO] Starting MBPO training loop for {self.cfg.max_iterations} iters...")

    for iteration in range(self.cfg.max_iterations):
      start_time = time.time()

      num_collected_steps = self.collect_data()
      collect_time = time.time() - start_time

      if self.replay_buffer.num_transitions < self.cfg.wm.horizon:
        print("[WARN] Insufficient data. Skipping world model and policy update.")
        continue

      start_wm_time = time.time()
      data = (
        self.replay_buffer.sample(self.cfg.wm.horizon, self.cfg.wm.batch_size)
        for _ in range(self.cfg.wm.num_batches)
      )
      wm_logs = self.world_model_learner.update(data)
      wm_learn_time = time.time() - start_wm_time

      start_policy_time = time.time()
      ppo_logs = {f"Policy/{k}": v for k, v in self.imagine().items()}
      policy_learn_time = time.time() - start_policy_time

      if iteration % self.cfg.log_interval == 0:
        self.logger.log(
          it=iteration,
          total_it=self.cfg.max_iterations,
          collect_time=collect_time,
          wm_learn_time=wm_learn_time,
          policy_learn_time=policy_learn_time,
          loss_dict={**wm_logs, **ppo_logs},
          num_collected_steps=num_collected_steps,
        )

      if (iteration + 1) % self.cfg.save_interval == 0:
        world_model_path = str(log_dir / "world_model.pt")
        actor_path = str(log_dir / "actor.pt")
        torch.save(self._wm.bundle(), world_model_path)
        torch.save(
          {
            "actor": self.alg.actor.state_dict(),
            "critic": self.alg.critic.state_dict(),
          },
          actor_path,
        )
        log_model_artifact([world_model_path, actor_path])

    self._cleanup()
