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

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim import ArticulationRootPropertiesCfg, RigidBodyPropertiesCfg, UsdFileCfg

from robotis_lab.assets.robots import ROBOTIS_LAB_ASSETS_DATA_DIR

FFW_SH5_CFG = ArticulationCfg(
    spawn=UsdFileCfg(
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
            "head_joint1": 0.5,
            "head_joint2": 0.0,

            # Lift joint
            "lift_joint": -0.15,
        },
    ),
    actuators={
        # Actuators for swerve base
        "base_steer": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_wheel_steer_joint",
                "right_wheel_steer_joint",
                "rear_wheel_steer_joint",
            ],
            velocity_limit_sim=10.0,
            effort_limit_sim=100000.0,
            stiffness=10000.0,
            damping=100.0,
        ),
        "base_drive": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_wheel_drive_joint",
                "right_wheel_drive_joint",
                "rear_wheel_drive_joint",
            ],
            velocity_limit_sim=30.0,
            effort_limit_sim=100000.0,
            stiffness=0.0,
            damping=500.0,
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

        # Actuators for hands
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[
                "finger_l_joint[1-9]",
                "finger_l_joint1[0-9]",
                "finger_l_joint20",
                "finger_r_joint[1-9]",
                "finger_r_joint1[0-9]",
                "finger_r_joint20",
            ],
            velocity_limit_sim=20.0,
            effort_limit_sim=1000000.0,
            stiffness=200.0,
            damping=10.0,
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
