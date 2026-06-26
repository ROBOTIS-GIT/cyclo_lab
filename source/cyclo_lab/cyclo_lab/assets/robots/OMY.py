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

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from cyclo_lab.assets.robots import CYCLO_LAB_ASSETS_DATA_DIR

OMY_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{CYCLO_LAB_ASSETS_DATA_DIR}/robots/OMY/OMY.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": -1.55,
            "joint3": 2.66,
            "joint4": -1.1,
            "joint5": 1.6,
            "joint6": 0.0,
            "rh_r1_joint": 0.0,
        },
    ),
    actuators={
        "DY_80": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-2]"],
            velocity_limit_sim=6.0,
            effort_limit_sim=61.4,
            stiffness=120.0,
            damping=4.0,
        ),
        "DY_70": ImplicitActuatorCfg(
            joint_names_expr=["joint[3-6]"],
            velocity_limit_sim=6.0,
            effort_limit_sim=31.7,
            stiffness=120.0,
            damping=4.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["rh_r1_joint", "rh_r2", "rh_l1", "rh_l2"],
            velocity_limit_sim=2.2,
            effort_limit_sim=30.0,
            stiffness=100.0,
            damping=4.0,
        ),
    },
)

"""Configuration of OMY arm using implicit actuator models."""
OMY_OFF_SELF_COLLISION_CFG = OMY_CFG.replace(
    spawn=OMY_CFG.spawn.replace(
        articulation_props=OMY_CFG.spawn.articulation_props.replace(
            enabled_self_collisions=False,
        )
    )
)
