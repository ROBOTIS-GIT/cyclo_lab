"""FFW SG2 ROS2-compatible topic and frame names."""

AI_WORKER_LEFT_ARM_TOPIC = "/leader/joint_trajectory_command_broadcaster_left/joint_trajectory"
AI_WORKER_RIGHT_ARM_TOPIC = "/leader/joint_trajectory_command_broadcaster_right/joint_trajectory"
HEAD_TOPIC = "/leader/joystick_controller_left/joint_trajectory"
LIFT_TOPIC = "/leader/joystick_controller_right/joint_trajectory"
CMD_VEL_TOPIC = "/cmd_vel"
JOINT_STATES_TOPIC = "/joint_states"
ODOM_TOPIC = "/odom"
TF_TOPIC = "/tf"
SIMULATION_RESET_TOPIC = "/simulation/reset"
BASE_FRAME = "base_link"
BASE_BODY = "world"
ODOM_FRAME = "odom"

FFW_SG2_ACTION_TOPICS = {
    "left_arm": AI_WORKER_LEFT_ARM_TOPIC,
    "right_arm": AI_WORKER_RIGHT_ARM_TOPIC,
    "head": HEAD_TOPIC,
    "lift": LIFT_TOPIC,
    "mobile": CMD_VEL_TOPIC,
}
FFW_SG2_JOYSTICK_TRIGGER_TOPIC = "/leader/joystick_controller/tact_trigger"
FFW_SG2_CAMERA_TOPICS = {
    "cam_head": "/zed/zed_node/left/image_rect_color/compressed",
    "cam_wrist_left": "/camera_left/camera_left/color/image_rect_raw/compressed",
    "cam_wrist_right": "/camera_right/camera_right/color/image_rect_raw/compressed",
}
