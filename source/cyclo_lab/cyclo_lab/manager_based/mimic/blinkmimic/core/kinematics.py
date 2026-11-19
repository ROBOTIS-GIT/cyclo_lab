"""Bridge BlinkMimic's matrix interface to the differentiable K1 URDF chain."""

from .rotations import mat_to_quat, quat_to_mat


class Kinematics:
    def __init__(self, bridge):
        self.chain = bridge.fk
        self.tracked_idx = range(len(self.chain.body_names))
        limits = bridge.robot.data.default_joint_pos_limits.torch[0]
        self.q_lo, self.q_hi = limits.unbind(-1)

    def forward(self, position, rotation, velocity, omega, joints, joint_velocity):
        p, q, v, w = self.chain(position, mat_to_quat(rotation), joints, velocity, omega, joint_velocity)
        return p, quat_to_mat(q), v, w
