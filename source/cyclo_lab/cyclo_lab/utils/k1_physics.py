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
from isaaclab_tasks.utils import PresetCfg, preset


def _newton_cfg(njmax: int, nconmax: int) -> NewtonCfg:
    return NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=njmax,
            nconmax=nconmax,
            cone="pyramidal",
            integrator="implicitfast",
            impratio=1.0,
            ls_iterations=50,
            ls_parallel=True,
        ),
        num_substeps=1,
        debug_mode=False,
    )


@configclass
class K1PhysicsCfg(PresetCfg):
    isaacsim_physx = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    physx = isaacsim_physx
    newton_mjwarp = _newton_cfg(njmax=512, nconmax=64)
    default = newton_mjwarp


@configclass
class K1LocomotionPhysicsCfg(K1PhysicsCfg):
    """Match the constraint allocation of the existing Newton walking task."""

    newton_mjwarp = _newton_cfg(njmax=95, nconmax=20)
    default = newton_mjwarp


def configure_k1_physics(env_cfg, *, locomotion: bool = False) -> None:
    """Declare backend differences before Isaac Lab resolves the selected presets."""
    from cyclo_lab.assets.robots.K1_rev1 import AI_SAPIENS_DESCRIPTION_DIR

    env_cfg.sim.physics = K1LocomotionPhysicsCfg() if locomotion else K1PhysicsCfg()
    env_cfg.sim.use_newton_actuators = True
    spawn = env_cfg.scene.robot.spawn
    spawn.self_collision = preset(default=True, isaacsim_physx=False, physx=False)
    spawn.ros_package_paths = preset(
        default=[{"name": "ai_sapiens_description", "path": AI_SAPIENS_DESCRIPTION_DIR}],
        isaacsim_physx=[], physx=[],
    )
    if locomotion:
        env_cfg.scene.contact_forces.force_threshold = preset(default=1.0, isaacsim_physx=None, physx=None)
        env_cfg.observations.critic.base_lin_vel.clip = preset(
            default=(-10.0, 10.0), isaacsim_physx=None, physx=None
        )
        # MJWarp takes the larger collider friction: let the robot material control contact.
        material = env_cfg.scene.terrain.physics_material
        material.static_friction = preset(default=0.0, isaacsim_physx=1.0, physx=1.0)
        material.dynamic_friction = preset(default=0.0, isaacsim_physx=1.0, physx=1.0)
        env_cfg.sim.physics_material = material
