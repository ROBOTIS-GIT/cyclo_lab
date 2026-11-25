# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in native simulator checks; never used as a synthetic training source."""

import argparse
import traceback
import warp as wp
import torch
from cyclo_lab.manager_based.mimic._shared.motion_bridge import BODIES, wxyz


def check_ik(runner, atol=5e-3):
    """Compare shared analytic IK to the live articulation in native joint order."""
    if runner.cfg.representation != "maxcoord":
        return {}
    bridge = runner.bridge
    q, qd = runner.rep.joint_state(runner.rep.represent(bridge.state())).chunk(2, -1)
    data = bridge.robot.data
    angle = q - data.joint_pos.torch
    angle = torch.atan2(angle.sin(), angle.cos())
    errors = {
        "joint_position_mod_2pi": float(angle.abs().max()),
        "joint_velocity": float((qd - data.joint_vel.torch).abs().max()),
    }
    if max(errors.values()) > atol:
        raise RuntimeError(f"Native articulation/IK mismatch: {errors}")
    return errors


def check_reset_twists(runner):
    """Write a rotated, moving state; verify link/COM readback, FK and optional IK.

    Reset the real environment after checking the written poses and velocities.
    """
    b = runner.bridge
    robot, env = b.robot, b.env
    n = env.num_envs
    ids = torch.arange(n, device=b.device)
    quat = torch.tensor([0.2, -0.3, 0.1, 1.0], device=b.device)
    quat = (quat / quat.norm()).expand(n, -1).clone()  # Isaac xyzw
    pos = env.scene.env_origins + torch.tensor([0.0, 0.0, 1.1], device=b.device)
    velocity = (
        torch.tensor([0.2, -0.1, 0.3, 0.4, -0.7, 0.9], device=b.device)
        .expand(n, -1)
        .clone()
    )
    joint_pos = robot.data.default_joint_pos.torch.clone()
    joint_vel = (
        torch.linspace(-0.3, 0.4, robot.num_joints, device=b.device)
        .expand(n, -1)
        .clone()
    )
    try:
        robot.write_joint_state_to_sim_index(
            position=joint_pos, velocity=joint_vel, env_ids=ids
        )
        robot.write_root_link_pose_to_sim_index(
            root_pose=torch.cat([pos, quat], -1), env_ids=ids
        )
        robot.write_root_link_velocity_to_sim_index(root_velocity=velocity, env_ids=ids)
        env.scene.write_data_to_sim()
        env.sim.forward()
        d = robot.data
        torch.testing.assert_close(d.root_link_pos_w.torch, pos, atol=2e-5, rtol=0)
        orientation_dot = (d.root_link_quat_w.torch * quat).sum(-1).abs()
        torch.testing.assert_close(
            orientation_dot, torch.ones_like(orientation_dot), atol=2e-5, rtol=0
        )
        actual = torch.cat(
            [d.root_link_lin_vel_w.torch, d.root_link_ang_vel_w.torch], -1
        )
        torch.testing.assert_close(actual, velocity, atol=2e-5, rtol=0)
        torch.testing.assert_close(d.joint_pos.torch, joint_pos, atol=2e-5, rtol=0)
        torch.testing.assert_close(d.joint_vel.torch, joint_vel, atol=2e-5, rtol=0)
        offset = d.root_com_pos_w.torch - d.root_link_pos_w.torch
        if not torch.all(offset.norm(dim=-1) > 1e-6):
            raise AssertionError("Rotating-pelvis check requires a nonzero COM offset")
        expected_com = velocity[:, :3] + torch.linalg.cross(velocity[:, 3:], offset)
        torch.testing.assert_close(
            d.root_com_lin_vel_w.torch, expected_com, atol=2e-5, rtol=0
        )
        return {
            "fk": check_fk(b, atol=2e-5),
            "ik": check_ik(runner, atol=2e-5),
            "com_offset_m": float(offset.norm(dim=-1).max()),
        }
    finally:
        env.reset()


def check_fk(bridge, atol=2e-3):
    state = bridge.generalized_state()
    pos, quat, vel, ang = bridge.fk(
        state[:, :3],
        state[:, 3:7],
        state[:, 7:30],
        state[:, 30:33],
        state[:, 33:36],
        state[:, 36:],
    )
    ids = [bridge.robot.body_names.index(n) for n in BODIES]
    d = bridge.robot.data
    actual_pos = d.body_link_pos_w.torch[:, ids] - bridge.env.scene.env_origins[:, None]
    actual_q = wxyz(d.body_link_quat_w.torch[:, ids])
    errors = {
        "position": (pos - actual_pos).abs().max(),
        "orientation": (1 - (quat * actual_q).sum(-1).abs()).abs().max(),
        "linear_velocity": (vel - bridge.tracked_linear_velocity()).abs().max(),
        "angular_velocity": (ang - d.body_link_ang_vel_w.torch[:, ids]).abs().max(),
    }
    if any(float(v) > atol for v in errors.values()):
        raise RuntimeError(f"Isaac/URDF FK mismatch: {errors}")
    return {k: float(v) for k, v in errors.items()}


def check_foot_friction(env, backend):
    robot = env.scene["robot"]
    bodies = robot.map_body_ids_to_backend(robot.find_bodies(".*_ankle_roll_link")[0])
    if backend == "newton_mjwarp":
        view = robot._root_view
        shapes = [shape for body in bodies for shape in view.body_shapes[body]]
        material = view.get_attribute("shape_material_mu", env.sim.physics_manager.get_model())[:, 0]
        coefficients = wp.to_torch(material)[:, shapes]
    else:
        view = robot.root_view
        shapes = []
        start = 0
        for index, path in enumerate(view.link_paths[0]):
            count = robot._physics_sim_view.create_rigid_body_view(path).max_shapes
            if index in bodies:
                shapes.extend(range(start, start + count))
            start += count
        coefficients = wp.to_torch(view.get_material_properties())[:, shapes, 0]
    assert shapes
    expected = env.tracking_friction.to(coefficients.device)
    torch.testing.assert_close(coefficients, expected.expand_as(coefficients))
    torso = robot.body_names.index("torso_link")
    torch.testing.assert_close(robot.data.body_com_pos_b.torch[:, torso], env.tracking_com)


def verify(backend, revision):
    import gymnasium as gym
    import cyclo_lab  # noqa: F401
    from isaaclab.app import launch_simulation
    from isaaclab_tasks.utils import resolve_task_config
    from cyclo_lab.manager_based.mimic.supermimic.training.runner import IsaacSuperMimicRunner

    task = f"Cyclo-SuperMimic-MaxCoord-K1-Rev{revision}-001"
    cfg, settings = resolve_task_config(task, "learner_cfg_entry_point", overrides=[f"physics={backend}"])
    cfg.scene.num_envs = 2
    settings.actor_hidden_dims = settings.dynamics_hidden_dims = settings.critic_hidden_dims = (32, 32)
    settings.eager = True
    with launch_simulation(cfg, {"headless": True, "require_kit": True}):
        env = gym.make(task, cfg=cfg)
        try:
            env.reset()
            check_foot_friction(env.unwrapped, backend)
            runner = IsaacSuperMimicRunner(env, settings)
            check_fk(runner.bridge, atol=2e-5)
            errors = check_reset_twists(runner)
            env.step(torch.zeros((cfg.scene.num_envs, runner.bridge.num_actions), device=runner.device))
            check_fk(runner.bridge, atol=2e-5)
            print(f"PASS {backend}: materials, FK, IK and reset velocities: {errors}", flush=True)
        except Exception:
            traceback.print_exc()
            raise
        finally:
            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("newton_mjwarp", "isaacsim_physx"))
    parser.add_argument("--revision", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    verify(args.backend, args.revision)
