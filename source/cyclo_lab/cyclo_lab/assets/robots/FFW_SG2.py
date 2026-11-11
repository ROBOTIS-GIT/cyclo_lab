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
# Author: Taehyeong Kim, Sunghyun Park

from copy import deepcopy

from cyclo_lab.assets.robots import CYCLO_LAB_ASSETS_DATA_DIR
from cyclo_lab.robot_specs.ffw.sg2.control import FFW_SG2_SWERVE_STEERING_ANGULAR_VELOCITY_LIMIT
from cyclo_lab.robot_specs.ffw.sg2.mobile_base import (
    SG2_SWERVE_DRIVE_DAMPING,
    SG2_SWERVE_STEERING_JOINTS,
    SG2_SWERVE_WHEEL_JOINTS,
)
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim import (
    ArticulationRootPropertiesCfg,
    RigidBodyPropertiesCfg,
    UsdFileCfg,
)

_SG2_PHYSICS_LIFT_EFFORT_LIMIT = 5_000_000.0
_SG2_PHYSICS_LIFT_STIFFNESS = 250_000.0
_SG2_PHYSICS_LIFT_DAMPING = 5_000.0


# Mass properties are stored in the USD; tasks can fix the root through their spawn configuration.
FFW_SG2_CFG = ArticulationCfg(
    spawn=UsdFileCfg(
        usd_path=f"{CYCLO_LAB_ASSETS_DATA_DIR}/robots/FFW/FFW_SG2.usd",
        rigid_props=RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # # Swerve base joints
            # "left_wheel_drive": 0.0, "left_wheel_steer": 0.0,
            # "right_wheel_drive": 0.0, "right_wheel_steer": 0.0,
            # "rear_wheel_drive": 0.0, "rear_wheel_steer": 0.0,

            **dict.fromkeys(SG2_SWERVE_STEERING_JOINTS + SG2_SWERVE_WHEEL_JOINTS, 0.0),
            # Left arm joints
            **{f"arm_l_joint{i + 1}": 0.0 for i in range(7)},
            # Right arm joints
            **{f"arm_r_joint{i + 1}": 0.0 for i in range(7)},
            # Left and right gripper joints
            **{f"gripper_l_joint{i + 1}": 0.0 for i in range(4)},
            **{f"gripper_r_joint{i + 1}": 0.0 for i in range(4)},
            # Head joints
            "head_joint1": 0.0,
            "head_joint2": 0.0,
            # Lift joint
            "lift_joint": 0.0,
        },
    ),
    actuators={

        # Simulation defaults from FFW_SH5.py (f4c0470, lines 217-229).
        # These gains and limits are not measured SG2 motor specifications.
        "base_steer": ImplicitActuatorCfg(
            joint_names_expr=list(SG2_SWERVE_STEERING_JOINTS),
            velocity_limit_sim=FFW_SG2_SWERVE_STEERING_ANGULAR_VELOCITY_LIMIT,
            effort_limit_sim=100000.0,
            stiffness=10000.0,
            damping=100.0,
        ),
        "base_drive": ImplicitActuatorCfg(
            joint_names_expr=list(SG2_SWERVE_WHEEL_JOINTS),
            velocity_limit_sim=50.0,
            effort_limit_sim=100000.0,
            # Track wheel velocity without pulling the wheel back to a fixed angle.
            stiffness=0.0,
            damping=SG2_SWERVE_DRIVE_DAMPING,
        ),
        # Actuator for vertical lift joint
        "lift": ImplicitActuatorCfg(
            joint_names_expr=["lift_joint"],
            velocity_limit_sim=0.2,
            effort_limit_sim=1000000.0,
            stiffness=10000.0,
            damping=100.0,
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
        # Actuators for grippers
        "gripper_master": ImplicitActuatorCfg(
            joint_names_expr=["gripper_l_joint1", "gripper_r_joint1"],
            velocity_limit_sim=2.2,
            effort_limit_sim=30.0,
            stiffness=100.0,
            damping=4.0,
        ),
        "gripper_passive": ImplicitActuatorCfg(
            joint_names_expr=["gripper_l_joint[2-4]", "gripper_r_joint[2-4]"],
            velocity_limit_sim=10.0,
            effort_limit_sim=1.0,
            stiffness=0.0,
            damping=0.0,
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


FFW_SG2_PHYSICS_CFG = deepcopy(FFW_SG2_CFG)
FFW_SG2_PHYSICS_CFG.spawn.rigid_props.linear_damping = 2.0
FFW_SG2_PHYSICS_CFG.spawn.rigid_props.angular_damping = 4.0
FFW_SG2_PHYSICS_CFG.articulation_root_prim_path = "/ffw_sg2_follower/world"
FFW_SG2_PHYSICS_CFG.actuators["lift"].effort_limit_sim = _SG2_PHYSICS_LIFT_EFFORT_LIMIT
FFW_SG2_PHYSICS_CFG.actuators["lift"].stiffness = _SG2_PHYSICS_LIFT_STIFFNESS
FFW_SG2_PHYSICS_CFG.actuators["lift"].damping = _SG2_PHYSICS_LIFT_DAMPING
