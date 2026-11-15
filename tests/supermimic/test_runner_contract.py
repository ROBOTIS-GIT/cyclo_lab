"""Unit doubles test the bridge contract; these are NOT Isaac runtime smoke tests."""

from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch
from cyclo_lab.learning.supermimic import ROOT
from cyclo_lab.learning.supermimic.runner import IsaacSuperMimicRunner, RunnerCfg


def proxy(value):
    return NS(torch=value)


class UnitEnv:
    def __init__(self):
        path = ROOT / "source/cyclo_lab/data/motions/K1_rev1/dance1/dance1.npz"
        with np.load(path) as d:
            arrays = {k: d[k].copy() for k in d.files}
        names = arrays["joint_names"].tolist()
        self.device = "cpu"
        self.spec = NS(id="Cyclo-SuperMimic-K1-Rev1-Dance1")
        self.num_envs = 1
        self.step_dt = 0.02
        self.unwrapped = self
        self.cfg = NS(
            decimation=4,
            commands=NS(reference_trajectory=NS(trajectory_file=str(path))),
        )
        self.cfg.terminations = NS(
            anchor_height=NS(params={"threshold": 0.25}),
            anchor_ori=NS(params={"threshold": 0.8}),
        )
        self.cfg.rewards = NS(
            **{
                n: NS(weight=w)
                for n, w in zip(
                    (
                        "reference_anchor_position",
                        "reference_anchor_orientation",
                        "reference_body_position",
                        "reference_body_orientation",
                        "reference_body_linear_velocity",
                        "reference_body_angular_velocity",
                    ),
                    (0.5, 0.5, 1.0, 1.0, 1.0, 1.0),
                )
            }
        )
        ref = NS(
            fps=[50],
            num_frames=len(arrays["joint_pos"]),
            joint_position=torch.tensor(arrays["joint_pos"]),
            joint_velocity=torch.tensor(arrays["joint_vel"]),
            _body_position_w=torch.tensor(arrays["body_pos_w"]),
            _body_orientation_w=torch.tensor(arrays["body_quat_w"]),
            _body_linear_velocity_w=torch.tensor(arrays["body_lin_vel_w"]),
            _body_angular_velocity_w=torch.tensor(arrays["body_ang_vel_w"]),
        )
        self.frames = torch.zeros(1, dtype=torch.long)
        self.resample_counts = torch.zeros(1, dtype=torch.long)
        self.command_manager = NS(
            get_term=lambda _: NS(
                reference=ref,
                frame_ids=self.frames,
                resample_counts=self.resample_counts,
            )
        )
        self.joints = torch.zeros(1, 23)
        data = NS(
            root_link_pos_w=proxy(torch.tensor([[10.0, 20.0, 0.8]])),
            root_link_quat_w=proxy(torch.tensor([[0.0, 0.0, 0.0, 1.0]])),
            root_link_lin_vel_w=proxy(torch.zeros(1, 3)),
            root_link_ang_vel_w=proxy(torch.zeros(1, 3)),
            joint_pos=proxy(self.joints),
            joint_vel=proxy(torch.zeros(1, 23)),
            default_joint_pos=proxy(torch.zeros(1, 23)),
            soft_joint_pos_limits=proxy(
                torch.tensor([[[-3.0, 3.0]]] * 23).reshape(1, 23, 2)
            ),
        )
        robot = NS(
            data=data,
            joint_names=names,
            body_names=arrays["body_names"].tolist(),
            num_joints=23,
        )

        class Scene(dict):
            pass

        self.scene = Scene(robot=robot)
        self.scene.env_origins = torch.tensor([[10.0, 20.0, 0.0]])
        self.action_manager = NS(
            action=torch.zeros(1, 23), get_term=lambda _: NS(_scale=0.1, _offset=0.0)
        )
        self.calls = 0

    def step(self, action):
        self.joints.add_(action.detach() * 0.01)
        self.calls += 1
        done = torch.tensor([self.calls % 4 == 0])
        self.frames.add_(1)
        self.frames[done] = 0
        self.resample_counts[done] += 1
        self.action_manager.action = action.detach().clone()
        self.action_manager.action[done] = 0
        return {}, torch.zeros(1), done, torch.zeros(1, dtype=torch.bool), {}


def test_collection_boundaries_learning_and_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "cyclo_lab.learning.supermimic.runner.environment_contract",
        lambda cfg, task, motion: {
            "unit_double": True,
            "task": task,
            "motion_sha256": motion["sha256"],
        },
    )
    before = torch._dynamo.config.disable
    torch._dynamo.config.disable = True
    try:
        env = UnitEnv()
        cfg = RunnerCfg(
            hidden=(16, 16),
            latent=8,
            batch_size=2,
            horizon=4,
            policy_horizon=2,
            eager=True,
        )
        runner = IsaacSuperMimicRunner(env, cfg)
        runner.collect(8)
        storage = runner.buffer._storage
        torch.testing.assert_close(
            storage["state"][0, 0, :3], torch.tensor([0.0, 0.0, 0.8])
        )
        torch.testing.assert_close(
            storage["state"][0, 0, 3:7], torch.tensor([1.0, 0.0, 0.0, 0.0])
        )
        assert storage["terminated"][0, 3] and storage["terminated"][0, 7]
        assert storage["motion_time"][0, 4] == 0
        assert storage["prev_action"][0, 4].count_nonzero() == 0
        for _ in range(5):
            logs = runner.update()
            assert logs["wm_parameters_changed"] and logs["actor_parameters_changed"]
            assert logs["wm_frozen_during_actor"]
        path = tmp_path / "checkpoint.pt"
        runner.save(path)
        saved = {k: v.clone() for k, v in runner.actor.state_dict().items()}
        fresh = IsaacSuperMimicRunner(UnitEnv(), cfg)
        fresh.load(path)
        assert fresh.iteration == 5
        for k, value in saved.items():
            torch.testing.assert_close(
                value, fresh.actor.state_dict()[k], rtol=0, atol=0
            )
        assert fresh.wm_learner.optimizers[0].state_dict()["state"]
        from cyclo_lab.learning.supermimic.checkpoint import assert_equal_tree

        assert_equal_tree(runner.state_bundle(), fresh.state_bundle())
        fresh.contract["motion_sha256"] = "wrong"
        with pytest.raises(ValueError, match="Incompatible"):
            fresh.load(path)
        bad = tmp_path / "ppo.pt"
        torch.save({"model_state_dict": {}}, bad)
        with pytest.raises(ValueError, match="Expected native Isaac"):
            fresh.load(bad)
    finally:
        torch._dynamo.config.disable = before


def test_clip_wrap_without_env_done_is_excluded_from_replay(monkeypatch):
    monkeypatch.setattr(
        "cyclo_lab.learning.supermimic.runner.environment_contract",
        lambda *args: {"unit_double": True},
    )

    class ClipEnv(UnitEnv):
        def step(self, action):
            obs, reward, done, timeout, extras = super().step(action)
            # Command-only wrap leaves the actual ActionManager history intact.
            self.action_manager.action = action.detach().clone()
            return obs, reward, torch.zeros_like(done), timeout, extras

    env = ClipEnv()
    runner = IsaacSuperMimicRunner(
        env,
        RunnerCfg(
            hidden=(16, 16),
            latent=8,
            batch_size=2,
            horizon=4,
            policy_horizon=2,
            eager=True,
        ),
    )
    runner.collect(12)
    storage = runner.buffer._storage
    assert not storage["terminated"].any()
    assert storage["truncated"][0, [3, 7, 11]].all()
    torch.testing.assert_close(
        storage["prev_action"][0, [4, 8]], storage["action"][0, [3, 7]], rtol=0, atol=0
    )
    for _ in range(10):
        sample = runner.buffer.sample(4, 16)
        assert not sample["truncated"][:, :-1].any()
        assert torch.equal(sample["motion_time"], torch.arange(4).expand(16, 4))
