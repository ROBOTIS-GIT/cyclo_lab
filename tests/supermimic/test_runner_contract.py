"""Unit doubles test the bridge contract; these are NOT Isaac runtime smoke tests."""

from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch
from cyclo_lab.utils.k1_motion_physics import URDF
from cyclo_lab.manager_based.mimic import REPO_ROOT as ROOT
from cyclo_lab.manager_based.mimic.supermimic.training.runner import WorldModelLearnerCfg, ActorCriticPolicyLearnerCfg
from cyclo_lab.manager_based.mimic.supermimic.training.runner import IsaacSuperMimicRunner, RunnerCfg


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
            foot_contact_z=0.074,
            scene=NS(robot=NS(spawn=NS(asset_path=str(URDF)))),
            sim=NS(physics=None),
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
                sampling_state=lambda: self.resample_counts.clone(),
                restore_sampling=self.resample_counts.copy_,
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
        data.default_joint_pos_limits = data.soft_joint_pos_limits
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
        from cyclo_lab.utils.k1_kinematics import UrdfKinematics

        self._fk = UrdfKinematics(URDF, names, robot.body_names)
        self._refresh_bodies()

    def _refresh_bodies(self):
        d = self.scene["robot"].data
        fields = self._fk(
            d.root_link_pos_w.torch,
            d.root_link_quat_w.torch[:, [3, 0, 1, 2]],
            d.joint_pos.torch,
            d.root_link_lin_vel_w.torch,
            d.root_link_ang_vel_w.torch,
            d.joint_vel.torch,
        )
        d.body_link_pos_w = proxy(fields[0])
        d.body_link_quat_w = proxy(fields[1][..., [1, 2, 3, 0]])
        d.body_link_lin_vel_w = proxy(fields[2])
        d.body_link_ang_vel_w = proxy(fields[3])

    def step(self, action):
        self.joints.add_(action.detach() * 0.01)
        self._refresh_bodies()
        self.calls += 1
        done = torch.tensor([self.calls % 4 == 0])
        self.frames.add_(1)
        self.frames[done] = 0
        self.resample_counts[done] += 1
        self.action_manager.action = action.detach().clone()
        self.action_manager.action[done] = 0
        return {}, torch.zeros(1), done, torch.zeros(1, dtype=torch.bool), {}


@pytest.mark.parametrize("representation", ["gencoord", "maxcoord"])
def test_collection_boundaries_learning_and_checkpoint(
    tmp_path, representation
):
    from cyclo_lab.manager_based.mimic._shared.checkpoint import save, load

    before = torch._dynamo.config.disable
    torch._dynamo.config.disable = True
    try:
        env = UnitEnv()
        cfg = RunnerCfg(
            representation=representation,
            actor_hidden_dims=(16, 16), dynamics_hidden_dims=(16, 16), critic_hidden_dims=(16, 16),
            latent_dim=8, eager=True,
            wm=WorldModelLearnerCfg(num_batches=1, batch_size=2, horizon=4),
            policy=ActorCriticPolicyLearnerCfg(batch_size=2, horizon=2),
        )
        runner = IsaacSuperMimicRunner(env, cfg)
        runner.collect(8)
        storage = runner.buffer._storage
        torch.testing.assert_close(
            storage["state"][0, 0, :3], torch.tensor([0.0, 0.0, 0.8])
        )
        quaternion_start = 3 if representation == "gencoord" else 42
        torch.testing.assert_close(
            storage["state"][0, 0, quaternion_start : quaternion_start + 4],
            torch.tensor([1.0, 0.0, 0.0, 0.0]),
        )
        assert storage["terminated"][0, 3] and storage["terminated"][0, 7]
        assert storage["motion_time"][0, 4] == 0
        assert storage["prev_action"][0, 4].count_nonzero() == 0
        wm_before = [p.detach().clone() for p in runner.dynamics.parameters()]
        actor_before = [p.detach().clone() for p in runner.actor.parameters()]
        for _ in range(5):
            metrics = runner.update()
            assert all(np.isfinite(value) for value in metrics.values())
        assert any(not torch.equal(a, b) for a, b in zip(wm_before, runner.dynamics.parameters()))
        assert any(not torch.equal(a, b) for a, b in zip(actor_before, runner.actor.parameters()))
        assert not any(p.requires_grad for p in runner.dynamics.parameters())
        save(runner, tmp_path, [])
        saved = {k: v.clone() for k, v in runner.actor.state_dict().items()}
        fresh = IsaacSuperMimicRunner(UnitEnv(), cfg)
        bundle = torch.load(tmp_path / "last.pt", weights_only=True)
        load(fresh, bundle, training=True)
        assert fresh.iteration == 5
        for k, value in saved.items():
            torch.testing.assert_close(
                value, fresh.actor.state_dict()[k], rtol=0, atol=0
            )
        assert fresh.wm_learner.optimizers[0].state_dict()["state"]
        for name in ("actor", "critic", "dynamics"):
            torch.testing.assert_close(getattr(runner, name).state_dict(), getattr(fresh, name).state_dict())
    finally:
        torch._dynamo.config.disable = before


def test_clip_wrap_without_env_done_is_excluded_from_replay():
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
            actor_hidden_dims=(16, 16), dynamics_hidden_dims=(16, 16), critic_hidden_dims=(16, 16),
            latent_dim=8, eager=True,
            wm=WorldModelLearnerCfg(num_batches=1, batch_size=2, horizon=4),
            policy=ActorCriticPolicyLearnerCfg(batch_size=2, horizon=2),
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


def test_blink_skips_nonfinite_actor_gradient_without_advancing_adam():
    from cyclo_lab.manager_based.mimic.blinkmimic.core.learner import PolicyLearner

    actor = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.Adam(actor.parameters())
    learner = PolicyLearner(
        NS(no_sync=False, pi_clip_value=0.1), actor, critic=None, dynamics=None,
        actor_optimizer=optimizer, critic_optimizer=None, mdp=None, motion=None,
        normalize_context=None, check_finite=True,
    )
    learner._update_actor(actor.weight.sum())
    before = actor.weight.detach().clone()
    adam_before = {key: value.clone() for key, value in optimizer.state[actor.weight].items()}
    loss_with_singular_gradient = torch.sqrt(actor.weight - before).sum()
    assert not learner._update_actor(loss_with_singular_gradient).isfinite()
    assert torch.equal(actor.weight, before)
    assert actor.weight.grad is None
    for key, value in adam_before.items():
        assert torch.equal(optimizer.state[actor.weight][key], value)
    assert learner._update_actor(actor.weight.sum()).isfinite()
