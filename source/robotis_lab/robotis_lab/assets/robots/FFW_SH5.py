# Copyright 2025 ROBOTIS CO., LTD.
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
# Author: Howon Kim

import re

from isaacsim.core.utils.stage import get_current_stage
from pxr import Sdf, UsdPhysics

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim import (
    ArticulationRootPropertiesCfg,
    RigidBodyMaterialCfg,
    RigidBodyPropertiesCfg,
    UsdFileCfg,
)
from isaaclab.sim.spawners.from_files import from_files
from isaaclab.sim.utils import bind_physics_material, clone, make_uninstanceable

from robotis_lab.assets.robots import ROBOTIS_LAB_ASSETS_DATA_DIR


_SH5_FINGER_TIP_MATERIAL = RigidBodyMaterialCfg(
    friction_combine_mode="max",
    restitution_combine_mode="min",
    static_friction=2.0,
    dynamic_friction=1.8,
    restitution=0.0,
)

_SH5_BASE_COLLISION_LINKS = (5, 6, 9, 10, 13, 14, 17, 18)
_SH5_WHEEL_DRIVE_LINKS = ("left_wheel_drive", "right_wheel_drive", "rear_wheel_drive")

SH5_SWERVE_STEERING_JOINTS = ("left_wheel_steer_joint", "right_wheel_steer_joint", "rear_wheel_steer_joint")
SH5_SWERVE_WHEEL_JOINTS = ("left_wheel_drive_joint", "right_wheel_drive_joint", "rear_wheel_drive_joint")
SH5_SWERVE_MODULE_X_OFFSETS = (0.1371, 0.1374, -0.289)
SH5_SWERVE_MODULE_Y_OFFSETS = (0.2554, -0.2554, 0.0)
SH5_SWERVE_MODULE_ANGLE_OFFSETS = (0.0, 0.0, 0.0)
SH5_SWERVE_WHEEL_RADIUS = 0.0865


def _is_sh5_finger_tip_prim(prim_path: str) -> bool:
    """Return true for SH5 finger link prims that need more grip."""
    path = prim_path.lower()
    return re.search(r"(^|/)finger_[rl]_link([1-9]|1[0-9]|20)(/|_|$)", path) is not None


def _filter_sh5_base_finger_collisions(stage, prim_path: str) -> None:
    """Disable collision checks between each hand base and the MCP/PIP finger links."""
    collision_paths_by_link_name = {}
    base_collision_paths_by_side = {"l": [], "r": []}
    for child_prim in stage.Traverse():
        child_path = str(child_prim.GetPath())
        if not child_path.startswith(f"{prim_path}/"):
            continue
        base_match = re.search(r"(^|/)hx5_([lr])_base/collisions(/|_|$)", child_path.lower())
        if base_match is not None:
            base_collision_paths_by_side[base_match.group(2)].append(child_path)
        if "/collisions/" not in child_path:
            continue
        match = re.search(r"(^|/)(finger_[lr]_link([1-9]|1[0-9]|20))(/|_|$)", child_path.lower())
        if match is not None:
            collision_paths_by_link_name.setdefault(match.group(2), []).append(child_path)

    for side in ("l", "r"):
        base_paths = base_collision_paths_by_side[side]
        finger_collision_paths = [
            finger_path
            for link_idx in _SH5_BASE_COLLISION_LINKS
            for finger_path in collision_paths_by_link_name.get(f"finger_{side}_link{link_idx}", [])
        ]
        for base_path in base_paths:
            base_prim = stage.GetPrimAtPath(base_path)
            filtered_pairs_api = UsdPhysics.FilteredPairsAPI.Apply(base_prim)
            filtered_pairs_rel = filtered_pairs_api.CreateFilteredPairsRel()
            for finger_path in finger_collision_paths:
                filtered_pairs_rel.AddTarget(Sdf.Path(finger_path))

        for finger_path in finger_collision_paths:
            finger_prim = stage.GetPrimAtPath(finger_path)
            filtered_pairs_api = UsdPhysics.FilteredPairsAPI.Apply(finger_prim)
            filtered_pairs_rel = filtered_pairs_api.CreateFilteredPairsRel()
            for base_path in base_paths:
                filtered_pairs_rel.AddTarget(Sdf.Path(base_path))

    filtered_sides = [side for side, base_paths in base_collision_paths_by_side.items() if base_paths]
    if filtered_sides:
        print("[SH5 collision filter] disabled hx5 base collision with MCP/PIP links.")


def _add_filtered_collision_pairs(stage, source_paths: list[str], target_paths: list[str]) -> None:
    for source_path in source_paths:
        source_prim = stage.GetPrimAtPath(source_path)
        filtered_pairs_api = UsdPhysics.FilteredPairsAPI.Apply(source_prim)
        filtered_pairs_rel = filtered_pairs_api.CreateFilteredPairsRel()
        for target_path in target_paths:
            filtered_pairs_rel.AddTarget(Sdf.Path(target_path))


def _filter_sh5_base_wheel_drive_collisions(stage, prim_path: str) -> None:
    """Disable collision checks between the base link and swerve drive wheel links."""
    base_collision_paths = []
    wheel_drive_collision_paths = []
    wheel_drive_pattern = "|".join(re.escape(link_name) for link_name in _SH5_WHEEL_DRIVE_LINKS)

    for child_prim in stage.Traverse():
        child_path = str(child_prim.GetPath())
        if not child_path.startswith(f"{prim_path}/") or "/collisions/" not in child_path:
            continue

        lower_path = child_path.lower()
        if re.search(r"(^|/)base_link/collisions(/|_|$)", lower_path):
            base_collision_paths.append(child_path)
        elif re.search(rf"(^|/)({wheel_drive_pattern})/collisions(/|_|$)", lower_path):
            wheel_drive_collision_paths.append(child_path)

    if not base_collision_paths or not wheel_drive_collision_paths:
        return

    _add_filtered_collision_pairs(stage, base_collision_paths, wheel_drive_collision_paths)
    _add_filtered_collision_pairs(stage, wheel_drive_collision_paths, base_collision_paths)
    print("[SH5 collision filter] disabled base_link collision with swerve drive wheel links.")


@clone
def spawn_sh5_with_finger_tip_friction(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn SH5 and bind high-friction material to fingertip bodies."""
    prim = from_files.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)

    material_path = f"{prim_path}/fingerTipPhysicsMaterial"
    _SH5_FINGER_TIP_MATERIAL.func(material_path, _SH5_FINGER_TIP_MATERIAL)

    stage = get_current_stage()
    make_uninstanceable(prim_path, stage)

    friction_prim_paths = set()
    for child_prim in stage.Traverse():
        child_path = str(child_prim.GetPath())
        if (
            child_path.startswith(f"{prim_path}/")
            and "/collisions/" in child_path
            and _is_sh5_finger_tip_prim(child_path)
        ):
            friction_prim_paths.add(child_path)

    for friction_prim_path in friction_prim_paths:
        bind_physics_material(friction_prim_path, material_path)

    _filter_sh5_base_finger_collisions(stage, prim_path)
    _filter_sh5_base_wheel_drive_collisions(stage, prim_path)

    return prim


FFW_SH5_CFG = ArticulationCfg(
    spawn=UsdFileCfg(
        func=spawn_sh5_with_finger_tip_friction,
        usd_path=f"{ROBOTIS_LAB_ASSETS_DATA_DIR}/robots/FFW/FFW_SH5.usd",
        rigid_props=RigidBodyPropertiesCfg(
            disable_gravity=True,
            linear_damping=2.0,
            angular_damping=4.0,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
        ),
        activate_contact_sensors=False,
    ),
    articulation_root_prim_path="/base_link/base_link",
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, -0.18),
        joint_pos={
            # Swerve base joints
            "left_wheel_drive_joint": 0.0, "left_wheel_steer_joint": 0.0,
            "right_wheel_drive_joint": 0.0, "right_wheel_steer_joint": 0.0,
            "rear_wheel_drive_joint": 0.0, "rear_wheel_steer_joint": 0.0,

            # Left arm joints
            **{f"arm_l_joint{i}": 0.0 for i in range(1, 8)},
            # Right arm joints
            **{f"arm_r_joint{i}": 0.0 for i in range(1, 8)},

            **{f"arm_l_joint{4}": -1.57},
            **{f"arm_r_joint{4}": -1.57},

            # Left and right hand joints
            **{f"finger_l_joint{i}": 0.0 for i in range(1, 21)},
            **{f"finger_r_joint{i}": 0.0 for i in range(1, 21)},

            # Head joints
            "head_joint1": 0.695,
            "head_joint2": 0.0,

            # Lift joint
            "lift_joint": 0.0,
        },
    ),
    actuators={
        # Actuators for swerve base
        "base_steer": ImplicitActuatorCfg(
            joint_names_expr=list(SH5_SWERVE_STEERING_JOINTS),
            velocity_limit_sim=10.0,
            effort_limit_sim=100000.0,
            stiffness=10000.0,
            damping=100.0,
        ),
        "base_drive": ImplicitActuatorCfg(
            joint_names_expr=list(SH5_SWERVE_WHEEL_JOINTS),
            velocity_limit_sim=50.0,
            effort_limit_sim=100000.0,
            stiffness=0.0,
            damping=100.0,
        ),

        # Actuator for vertical lift joint
        "lift": ImplicitActuatorCfg(
            joint_names_expr=["lift_joint"],
            velocity_limit_sim=0.2,  # 0.0001
            effort_limit_sim=1000000.0,
            stiffness=10000.0,
            damping=100.0,  # 0.0
        ),

        # Actuators for both arms
        "DY_80": ImplicitActuatorCfg(
            joint_names_expr=[
                "arm_l_joint[1-2]",
                "arm_r_joint[1-2]",
            ],
            velocity_limit_sim=15.0,
            effort_limit_sim=61.4,
            stiffness=600.0,
            damping=30.0,
        ),
        "DY_70": ImplicitActuatorCfg(
            joint_names_expr=[
                "arm_l_joint[3-6]",
                "arm_r_joint[3-6]",
            ],
            velocity_limit_sim=15.0,
            effort_limit_sim=31.7,
            stiffness=600.0,
            damping=20.0,
        ),
        "DP-42": ImplicitActuatorCfg(
            joint_names_expr=[
                "arm_l_joint7",
                "arm_r_joint7",
            ],
            velocity_limit_sim=6.0,
            effort_limit_sim=5.1,
            stiffness=200.0,
            damping=3.0,
        ),

        # Actuators for hands
        "XM_335": ImplicitActuatorCfg(
            joint_names_expr=[
                "finger_l_joint[1-9]",
                "finger_l_joint1[0-9]",
                "finger_l_joint20",
                "finger_r_joint[1-9]",
                "finger_r_joint1[0-9]",
                "finger_r_joint20",
            ],
            velocity_limit_sim=30.0,
            effort_limit_sim=5.15,    
            stiffness=600.0,   
            damping=20.0,
        ),

        # Actuators for head joints
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_joint1", "head_joint2"],
            velocity_limit_sim=2.0,
            effort_limit_sim=30.0,
            stiffness=150.0,
            damping=3.0,
        ),
    },
)
