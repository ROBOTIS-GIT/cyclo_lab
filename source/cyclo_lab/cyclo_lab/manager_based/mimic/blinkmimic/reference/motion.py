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

"""Use the same name-reordered, link-origin reference as the Isaac environment."""

import torch

from ..core.motion import MotionLib
from ..core.rotations import quat_to_mat


class IsaacMotionLib(MotionLib):
    def __init__(self, bridge):
        reference = bridge.motion
        self.root_p = reference.root_pos_w
        self.root_R = quat_to_mat(reference.root_quat_w)
        self.q, self.qd = reference.joint_pos, reference.joint_vel
        self.root_v, self.root_w = reference.root_lin_vel_w, reference.root_ang_vel_w
        inverse = self.root_R.transpose(-1, -2)
        self.root_v_local = (inverse @ self.root_v[..., None]).squeeze(-1)
        self.root_w_local = (inverse @ self.root_w[..., None]).squeeze(-1)
        frames = torch.arange(len(self.q), device=self.q.device)
        self.clip_end = reference.motion_ends - 1
        self.names = [str(i) for i in range(len(self.clip_end))]
        self.clip_id = torch.bucketize(frames, reference.motion_ends, right=True)
        p, q, v, w = reference.body_state(frames)
        self.body_p, self.body_R, self.body_v, self.body_w = p, quat_to_mat(q), v, w
