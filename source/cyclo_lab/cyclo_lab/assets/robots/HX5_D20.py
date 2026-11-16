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
# Author: Sunghyun Park

"""Configurations for the standalone ROBOTIS HX5-D20 (rev2) left and right hands.

The USD files carry the joint limits, masses and the official self-collision filters, and fix the hand base to the
world with a fixed joint (root_joint). The actuator values below are the joint drives stored in the USD.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from cyclo_lab.assets.robots import CYCLO_LAB_ASSETS_DATA_DIR

HX5_D20_LEFT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{CYCLO_LAB_ASSETS_DATA_DIR}/robots/HAND/HX5_D20_LEFT.usd",
        # True keeps the hand base fixed to the world. Set False to free the base, e.g. to attach the hand to an arm.
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={"finger_l_joint.*": 0.0},
    ),
    actuators={
        "XM_335": ImplicitActuatorCfg(
            joint_names_expr=["finger_l_joint.*"],
            joint_velocity_limit=4.8,  # robotis_hand URDF joint limit
            joint_effort_limit=1.03,  # XM335-T323-T stall torque at 11.1 V
            stiffness=500.0,  # same as the FFW-SH5 hand actuators
            damping=30.0,
        ),
    },
)
"""Standalone HX5-D20 left hand."""

HX5_D20_RIGHT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{CYCLO_LAB_ASSETS_DATA_DIR}/robots/HAND/HX5_D20_RIGHT.usd",
        # True keeps the hand base fixed to the world. Set False to free the base, e.g. to attach the hand to an arm.
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={"finger_r_joint.*": 0.0},
    ),
    actuators={
        "XM_335": ImplicitActuatorCfg(
            joint_names_expr=["finger_r_joint.*"],
            joint_velocity_limit=4.8,  # robotis_hand URDF joint limit
            joint_effort_limit=1.03,  # XM335-T323-T stall torque at 11.1 V
            stiffness=500.0,  # same as the FFW-SH5 hand actuators
            damping=30.0,
        ),
    },
)
"""Standalone HX5-D20 right hand."""
