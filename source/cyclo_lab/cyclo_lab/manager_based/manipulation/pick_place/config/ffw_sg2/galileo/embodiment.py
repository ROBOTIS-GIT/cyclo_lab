"""FFW-SG2 embodiment used by the Galileo Arena task."""

from __future__ import annotations

from copy import deepcopy

import isaaclab.envs.mdp as mdp
import torch
from cyclo_lab.assets.robots import FFW_SG2_PHYSICS_CFG
from cyclo_lab.assets.sensors.ffw_sg2_cameras import make_ffw_sg2_head_camera_cfg, make_ffw_sg2_wrist_camera_cfg
from cyclo_lab.manager_based.actions import FFWSG2MobileActionsCfg
from cyclo_lab.robot_specs.ffw.sg2 import FFW_SG2_PUBLISHED_JOINT_NAMES
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase


def _make_robot_cfg() -> ArticulationCfg:
    robot_cfg = deepcopy(FFW_SG2_PHYSICS_CFG).replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot_cfg.spawn.rigid_props.disable_gravity = False
    return robot_cfg


def _base_twist(env, asset_name: str) -> torch.Tensor:
    robot = env.scene[asset_name]
    return torch.cat((robot.data.root_lin_vel_b[:, :2], robot.data.root_ang_vel_b[:, 2:3]), dim=-1)


@configclass
class FFWSG2GalileoSceneCfg:
    """SG2 articulation and end-effector frames required by Arena tasks."""

    robot: ArticulationCfg = _make_robot_cfg()
    left_ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ffw_sg2_follower/arm_base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/ffw_sg2_follower/arm_l_link7",
                name="left_end_effector",
                offset=OffsetCfg(pos=(0.0, 0.0, -0.2)),
            )
        ],
    )
    right_ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ffw_sg2_follower/arm_base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/ffw_sg2_follower/arm_r_link7",
                name="right_end_effector",
                offset=OffsetCfg(pos=(0.0, 0.0, -0.2)),
            )
        ],
    )


@configclass
class FFWSG2GalileoCameraCfg:
    cam_head: CameraCfg = make_ffw_sg2_head_camera_cfg(height=480, width=640)
    cam_wrist_left: CameraCfg = make_ffw_sg2_wrist_camera_cfg("left")
    cam_wrist_right: CameraCfg = make_ffw_sg2_wrist_camera_cfg("right")


@configclass
class FFWSG2GalileoObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        robot_joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=list(FFW_SG2_PUBLISHED_JOINT_NAMES),
                    preserve_order=True,
                )
            },
        )
        robot_joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=list(FFW_SG2_PUBLISHED_JOINT_NAMES),
                    preserve_order=True,
                )
            },
        )
        base_twist = ObsTerm(func=_base_twist, params={"asset_name": "robot"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@register_asset
class FFWSG2GalileoEmbodiment(EmbodimentBase):
    """FFW-SG2 with 19 absolute joint targets and 3D swerve velocity."""

    name = "ffw_sg2_mobile_abs_joint_pos"
    default_arm_mode = ArmMode.DUAL_ARM

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose=None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
    ) -> None:
        super().__init__(enable_cameras, initial_pose, concatenate_observation_terms, arm_mode)
        self.scene_config = FFWSG2GalileoSceneCfg()
        self.camera_config = FFWSG2GalileoCameraCfg()
        self.action_config = FFWSG2MobileActionsCfg()
        self.observation_config = FFWSG2GalileoObservationsCfg()
        self.observation_config.policy.concatenate_terms = concatenate_observation_terms

    def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
        if arm_mode == ArmMode.LEFT:
            return "left_ee_frame"
        if arm_mode == ArmMode.RIGHT:
            return "right_ee_frame"
        raise ValueError("An individual arm is required for an SG2 end-effector frame.")

    def get_command_body_name(self) -> str:
        return "arm_r_link7"
