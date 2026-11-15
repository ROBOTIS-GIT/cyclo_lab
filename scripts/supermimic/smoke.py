# SPDX-License-Identifier: Apache-2.0
"""Bounded simulator + differentiable world-model construction check; no training."""
import argparse
import sys
from pathlib import Path
import torch
from k1 import ROOT, TASK, register

sys.path.insert(0, str(ROOT / 'third_party/supermimic/src'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion-file', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--learn', action='store_true', help='Also perform one tiny world-model and policy update')
    args = parser.parse_args()
    register()
    from mjlab.tasks.registry import load_env_cfg
    from supermimic.scripts.train_utils import create_env
    from supermimic.runners.supermimic_runner import BaseRunnerCfg, build_world_model
    cfg = load_env_cfg(TASK, play=True)
    cfg.commands['motion'].motion_file = str(args.motion_file.resolve())
    env = create_env(cfg, num_envs=1, device=args.device)
    try:
        wm = build_world_model(BaseRunnerCfg(dynamics_hidden_dims=(32, 32)), env, args.device)
        assert wm.num_actions == 23
        assert wm.representation.full_state_dim == 59
        for _ in range(3):
            obs, reward, done, extras = env.step(torch.zeros((1, 23), device=args.device))
            assert torch.isfinite(reward).all()
        print('PASS: K1 environment, 23 actions, 59-state world model, 3 simulator steps')
        if args.learn:
            from supermimic.runners.supermimic_runner import SuperMimicRunner, SuperMimicRunnerCfg
            from supermimic.learners.wm_learner import WorldModelLearnerCfg
            from supermimic.learners.policy_learner import ActorCriticPolicyLearnerCfg
            runner_cfg = SuperMimicRunnerCfg(
                dynamics_hidden_dims=(32, 32), actor_hidden_dims=(32, 32),
                critic_hidden_dims=(32, 32), latent_dim=8, buffer_size=32,
                sampling_mode='start',
                wm=WorldModelLearnerCfg(num_batches=1, batch_size=2, horizon=2),
                policy=ActorCriticPolicyLearnerCfg(num_batches=1, batch_size=2, horizon=2))
            runner = SuperMimicRunner(env, runner_cfg, TASK, device=args.device)
            runner._seed_sim_history()
            for _ in range(12):
                runner.step()
            wm_before = [p.detach().clone() for p in runner._dynamics.parameters()]
            actor_before = [p.detach().clone() for p in runner.policy_learner.actor.parameters()]
            batch = runner.replay_buffer.sample(2, 2)
            wm_logs = runner.world_model_learner.update(iter([batch]))
            policy_logs = runner.policy_learner.update()
            for value in {**wm_logs, **policy_logs}.values():
                assert torch.isfinite(torch.as_tensor(value)).all()
            assert any(not torch.equal(a, b) for a, b in zip(wm_before, runner._dynamics.parameters()))
            assert any(not torch.equal(a, b) for a, b in zip(actor_before, runner.policy_learner.actor.parameters()))
            print('PASS: one real world-model update and one policy update; parameters changed; all metrics finite')
    finally:
        env.close()
