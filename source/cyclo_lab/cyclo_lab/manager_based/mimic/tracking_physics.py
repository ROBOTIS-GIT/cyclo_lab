"""Apply K1 tracking randomization and expose the same sampled dynamics to the learner."""

import torch
import warp as wp


def write_friction(env, robot, friction, feet_only):
    """One coefficient per environment, shared by all selected collision shapes."""
    body_ids = robot.find_bodies(".*_ankle_roll_link")[0] if feet_only else None
    if type(env.cfg.sim.physics).__module__.startswith("isaaclab_newton."):
        from newton import ModelFlags

        manager = env.sim.physics_manager
        view = robot._root_view
        materials = wp.to_torch(view.get_attribute("shape_material_mu", manager.get_model())[:, 0])
        shapes = (
            slice(None)
            if body_ids is None
            else [
                shape for body in robot.map_body_ids_to_backend(body_ids) for shape in view.body_shapes[body]
            ]
        )
        if body_ids is not None and not shapes:
            raise ValueError("No Newton contact shapes on the selected feet")
        materials[:, shapes] = friction
        manager.add_model_change(ModelFlags.SHAPE_PROPERTIES)
        return

    view = robot.root_view
    materials = wp.to_torch(view.get_material_properties())
    shapes = slice(None)
    if body_ids is not None:
        selected = set(robot.map_body_ids_to_backend(body_ids))
        shapes = []
        offset = 0
        for body, path in enumerate(view.link_paths[0]):
            count = robot._physics_sim_view.create_rigid_body_view(path).max_shapes
            if body in selected:
                shapes.extend(range(offset, offset + count))
            offset += count
        if not shapes or offset != view.max_shapes:
            raise ValueError("PhysX link shapes do not match the articulation material view")
    materials[:, shapes, :2] = friction.to(materials.device)[:, :, None]
    ids = torch.arange(env.num_envs, dtype=torch.int32, device=materials.device)
    view.set_material_properties(wp.from_torch(materials), wp.from_torch(ids))


def initialize_dynamics(env, env_ids, feet_only, evaluation=False):
    robot = env.scene["robot"]
    torso = robot.body_names.index("torso_link")
    is_newton = type(env.cfg.sim.physics).__module__.startswith("isaaclab_newton.")
    poses = (robot.data.body_com_pos_b if is_newton else robot.data.body_com_pose_b).torch.clone()
    nominal = poses[:, torso, :3].clone()
    friction = torch.ones_like(nominal[:, :1]).contiguous()
    extent = nominal.new_tensor([0.025, 0.05, 0.05])
    if not evaluation:
        friction.uniform_(0.3, 1.2)
        poses[:, torso, :3] += torch.empty_like(nominal).uniform_(-1, 1) * extent
        robot.set_coms_index(
            coms=poses, env_ids=torch.arange(env.num_envs, device=env.device, dtype=torch.int32)
        )
    write_friction(env, robot, friction, feet_only)
    actual = robot.data.body_com_pos_b.torch[:, torso].clone()
    env.tracking_friction = friction
    env.tracking_com = actual
    env.tracking_com_delta = actual - nominal
    term = env.action_manager.get_term("joint_pos")
    env.nominal_action_offset = term._offset.clone()
    env.encoder_bias = torch.zeros_like(env.nominal_action_offset)
    if feet_only and not evaluation:
        sample_encoder_bias(env, slice(None))


def sample_encoder_bias(env, env_ids):
    if env_ids is None:
        env_ids = slice(None)
    bias = torch.empty_like(env.encoder_bias[env_ids]).uniform_(-0.01, 0.01)
    env.encoder_bias[env_ids] = bias
    env.action_manager.get_term("joint_pos")._offset[env_ids] = env.nominal_action_offset[env_ids] - bias
