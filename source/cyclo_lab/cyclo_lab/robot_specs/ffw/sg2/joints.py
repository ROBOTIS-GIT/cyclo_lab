"""FFW SG2 joint groups and public joint orders."""

FFW_SG2_LEFT_ARM_JOINT_NAMES = tuple(f"arm_l_joint{index}" for index in range(1, 8))
FFW_SG2_RIGHT_ARM_JOINT_NAMES = tuple(f"arm_r_joint{index}" for index in range(1, 8))
FFW_SG2_LEFT_GRIPPER_JOINT_NAMES = ("gripper_l_joint1",)
FFW_SG2_RIGHT_GRIPPER_JOINT_NAMES = ("gripper_r_joint1",)
FFW_SG2_HEAD_JOINT_NAMES = ("head_joint1", "head_joint2")
FFW_SG2_LIFT_JOINT_NAME = "lift_joint"
FFW_SG2_LIFT_JOINT_NAMES = (FFW_SG2_LIFT_JOINT_NAME,)

# ROS joint_states order. This matches the real robot observation surface and
# keeps mimic gripper joints filtered out.
FFW_SG2_PUBLISHED_JOINT_NAMES = (
    *FFW_SG2_LEFT_ARM_JOINT_NAMES,
    *FFW_SG2_LEFT_GRIPPER_JOINT_NAMES,
    *FFW_SG2_RIGHT_ARM_JOINT_NAMES,
    *FFW_SG2_RIGHT_GRIPPER_JOINT_NAMES,
    *FFW_SG2_HEAD_JOINT_NAMES,
    *FFW_SG2_LIFT_JOINT_NAMES,
)

# Isaac Lab action tensor order for Cyclo-Real-Pick-Place-FFW-SG2-v0.
# The ActionCfg dataclass declares lift before head, so this must stay separate
# from the joint_states publication order above.
FFW_SG2_ACTION_JOINT_NAMES = (
    *FFW_SG2_LEFT_ARM_JOINT_NAMES,
    *FFW_SG2_LEFT_GRIPPER_JOINT_NAMES,
    *FFW_SG2_RIGHT_ARM_JOINT_NAMES,
    *FFW_SG2_RIGHT_GRIPPER_JOINT_NAMES,
    *FFW_SG2_LIFT_JOINT_NAMES,
    *FFW_SG2_HEAD_JOINT_NAMES,
)

# Index map for converting tensors from the ROS publication order to the
# Isaac Lab action order without duplicating order assumptions at call sites.
FFW_SG2_PUBLISHED_TO_ACTION_INDICES = tuple(
    FFW_SG2_PUBLISHED_JOINT_NAMES.index(name) for name in FFW_SG2_ACTION_JOINT_NAMES
)

# Physical position limits from the shipped SG2 articulation. Topic bridges use
# this table as the final safety boundary for absolute joint targets.
FFW_SG2_JOINT_POSITION_LIMITS = {
    "arm_l_joint1": (-3.14, 3.14),
    "arm_l_joint2": (0.0, 3.14),
    "arm_l_joint3": (-3.14, 3.14),
    "arm_l_joint4": (-2.9361, 1.0786),
    "arm_l_joint5": (-3.14, 3.14),
    "arm_l_joint6": (-1.57, 1.57),
    "arm_l_joint7": (-1.8201, 1.5804),
    "gripper_l_joint1": (0.0, 1.1),
    "arm_r_joint1": (-3.14, 3.14),
    "arm_r_joint2": (-3.14, 0.0),
    "arm_r_joint3": (-3.14, 3.14),
    "arm_r_joint4": (-2.9361, 1.0786),
    "arm_r_joint5": (-3.14, 3.14),
    "arm_r_joint6": (-1.57, 1.57),
    "arm_r_joint7": (-1.5804, 1.8201),
    "gripper_r_joint1": (0.0, 1.1),
    "lift_joint": (-0.5, 0.0),
    "head_joint1": (-0.2317, 0.6951),
    "head_joint2": (-0.35, 0.35),
}
