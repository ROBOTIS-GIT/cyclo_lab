"""FFW-SH5 camera sensors backed by camera prims authored in the robot USD."""

from __future__ import annotations

FFW_SH5_HEAD_CAMERA_NAME = "cam_head"
FFW_SH5_WRIST_LEFT_CAMERA_NAME = "cam_wrist_left"
FFW_SH5_WRIST_RIGHT_CAMERA_NAME = "cam_wrist_right"

FFW_SH5_CAMERA_HEIGHT = 480
FFW_SH5_CAMERA_WIDTH = 640
FFW_SH5_WRIST_IMAGE_ROTATION_QUARTER_TURNS = 3

FFW_SH5_CAMERA_IMAGE_ROTATIONS = {
    FFW_SH5_WRIST_LEFT_CAMERA_NAME: FFW_SH5_WRIST_IMAGE_ROTATION_QUARTER_TURNS,
    FFW_SH5_WRIST_RIGHT_CAMERA_NAME: FFW_SH5_WRIST_IMAGE_ROTATION_QUARTER_TURNS,
}

_FFW_SH5_CAMERA_PRIM_PATHS = {
    FFW_SH5_HEAD_CAMERA_NAME: "{ENV_REGEX_NS}/Robot/base_link/head_link2/zed/Head_Camera",
    FFW_SH5_WRIST_LEFT_CAMERA_NAME: (
        "{ENV_REGEX_NS}/Robot/base_link/arm_l_link7/camera_l_bottom_screw_frame/camera_l_link/Left_Camera"
    ),
    FFW_SH5_WRIST_RIGHT_CAMERA_NAME: (
        "{ENV_REGEX_NS}/Robot/base_link/arm_r_link7/camera_r_bottom_screw_frame/camera_r_link/Right_Camera"
    ),
}


def make_ffw_sh5_camera_cfg(
    camera_name: str,
    *,
    update_period: float = 0.0,
    height: int = FFW_SH5_CAMERA_HEIGHT,
    width: int = FFW_SH5_CAMERA_WIDTH,
):
    """Register one existing SH5 USD camera prim as an IsaacLab RGB sensor."""
    from isaaclab.sensors import CameraCfg

    try:
        prim_path = _FFW_SH5_CAMERA_PRIM_PATHS[camera_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported FFW-SH5 camera: {camera_name}") from exc

    return CameraCfg(
        prim_path=prim_path,
        update_period=update_period,
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=None,
    )
