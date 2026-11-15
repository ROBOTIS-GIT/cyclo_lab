# Copyright 2026 ROBOTIS CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in native simulator checks; never used as a synthetic training source."""

import inspect
import torch


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

    Reset the real environment afterward. This diagnostic is called only by
    --verify, before collection or playback, and does not supply replay data.
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
            "fk": b.check_fk(atol=2e-5),
            "ik": check_ik(runner, atol=2e-5),
            "com_offset_m": float(offset.norm(dim=-1).max()),
        }
    finally:
        env.reset()


def runtime_physics(env):
    """Read the instantiated manager/solver, including PhysX's authored scene API."""
    env = env.unwrapped
    manager = env.sim.physics_manager
    if isinstance(manager, str):
        from isaaclab.utils.string import string_to_callable

        manager = string_to_callable(manager)
    solver_source = None
    if manager.__module__.startswith("isaaclab_newton."):
        solver = getattr(manager, "_solver", None)
        if solver is None or type(solver).__name__ != "SolverMuJoCo":
            raise RuntimeError("Expected an initialized native Newton/MJWarp solver")
        solver_name = f"{type(solver).__module__}.{type(solver).__name__}"
        solver_source = inspect.getsourcefile(type(solver))
    elif manager.__module__.startswith("isaaclab_physx."):
        scene = env.sim.stage.GetPrimAtPath(env.cfg.sim.physics_prim_path)
        solver_name = scene.GetAttribute("physxScene:solverType").Get()
        expected = "TGS" if env.cfg.sim.physics.solver_type == 1 else "PGS"
        if solver_name != expected:
            raise RuntimeError(
                f"Instantiated PhysX solver mismatch: {solver_name} != {expected}"
            )
    else:
        raise RuntimeError(f"Unexpected native physics manager: {manager}")
    return {
        "manager": f"{manager.__module__}.{manager.__name__}",
        "manager_source": inspect.getsourcefile(manager),
        "solver": solver_name,
        "solver_source": solver_source,
        "sim_dt": env.cfg.sim.dt,
        "decimation": env.cfg.decimation,
        "newton_num_substeps": getattr(env.cfg.sim.physics, "num_substeps", None),
        "representation_actuation_order": list(env.scene["robot"].joint_names),
    }


def check_motion_wrap(runner):
    """Force the final reference frame and observe a real command-only boundary.

    This diagnostic transition is excluded from replay. The separate collector
    regression verifies that this same resample counter marks replay truncation.
    """
    b = runner.bridge
    env, robot = b.env, b.robot
    command = env.command_manager.get_term("reference_trajectory")
    ids = torch.arange(env.num_envs, device=b.device)
    try:
        command.frame_ids.fill_(command.reference.num_frames - 1)
        before = command.resample_counts.clone()
        robot.write_joint_state_to_sim_index(
            position=command.joint_pos.clone(),
            velocity=command.joint_vel.clone(),
            env_ids=ids,
        )
        robot.write_root_link_pose_to_sim_index(
            root_pose=torch.cat(
                [command.body_pos_w[:, 0], command.body_quat_w[:, 0]], -1
            ),
            env_ids=ids,
        )
        robot.write_root_link_velocity_to_sim_index(
            root_velocity=torch.cat(
                [
                    b.motion.root_lin_vel_w[command.frame_ids],
                    b.motion.root_ang_vel_w[command.frame_ids],
                ],
                -1,
            ),
            env_ids=ids,
        )
        env.scene.write_data_to_sim()
        env.sim.forward()
        scale, offset = b.action_transform()
        action = (command.joint_pos - offset) / scale
        _, _, terminated, truncated, _ = env.step(action)
        if terminated.any() or truncated.any():
            raise AssertionError(
                f"Expected command-only wrap, got env done flags: {terminated}, {truncated}"
            )
        if not torch.all(command.resample_counts == before + 1):
            raise AssertionError(
                "Final-frame step did not produce exactly one command resample"
            )
        return {
            "pre_step_frame": command.reference.num_frames - 1,
            "resampled_envs": env.num_envs,
            "terminated": False,
            "truncated": False,
        }
    finally:
        env.reset()
