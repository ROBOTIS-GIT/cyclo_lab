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

"""Physics presets shared by K1 locomotion and motion tracking."""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab.utils import configclass
from isaaclab_tasks.utils import PresetCfg


@configclass
class K1PhysicsCfg(PresetCfg):
    isaacsim_physx = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    physx = isaacsim_physx
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=512,
            nconmax=64,
            cone="pyramidal",
            integrator="implicitfast",
            impratio=1.0,
            ls_iterations=50,
            ls_parallel=True,
        ),
        num_substeps=1,
        debug_mode=False,
    )
    default = newton_mjwarp
