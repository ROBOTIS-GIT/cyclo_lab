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
# Author: Taehyeong Kim

from isaaclab.sim import UsdFileCfg, RigidBodyPropertiesCfg, ArticulationRootPropertiesCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

from cyclo_lab.assets.robots import CYCLO_LAB_ASSETS_DATA_DIR

FFW_SG2_CFG = ArticulationCfg(
    spawn=UsdFileCfg(
        usd_path=f"{CYCLO_LAB_ASSETS_DATA_DIR}/robots/FFW/FFW_SG2.usd",
        rigid_props=RigidBodyPropertiesCfg(
            disable_gravity=True,
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
        # Actuators for swerve base
        # "base": ImplicitActuatorCfg(
        #     joint_names_expr=[
        #         "left_wheel_drive", "left_wheel_steer",
        #         "right_wheel_drive", "right_wheel_steer",
        #         "rear_wheel_drive", "rear_wheel_steer",
        #     ],
        #     velocity_limit_sim=30.0,
        #     effort_limit_sim=100000.0,
        #     stiffness=10000.0,
        #     damping=100.0,
        # ),

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
        "DP-42" : ImplicitActuatorCfg(
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
        "gripper_slave": ImplicitActuatorCfg(
            joint_names_expr=["gripper_l_joint[2-4]", "gripper_r_joint[2-4]"],
            effort_limit_sim=20.0,
            stiffness=2.0,
            damping=0.5,
        ),

        # Actuators for head joints
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_joint1", "head_joint2"],
            velocity_limit_sim=2.0,
            effort_limit_sim=30.0,
            stiffness=150.0,
            damping=3.0,
        ),
    }
)
