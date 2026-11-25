# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Kiwoong Park

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
