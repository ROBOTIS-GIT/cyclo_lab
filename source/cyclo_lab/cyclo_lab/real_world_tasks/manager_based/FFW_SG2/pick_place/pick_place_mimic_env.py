# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from collections.abc import Sequence

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv, ManagerBasedRLEnvCfg


class FFWSG2PickPlaceMimicEnv(ManagerBasedRLMimicEnv):
    """
    Isaac Lab Mimic environment wrapper class for FFW SG2 Pick and Place.
    """

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot_root_pos = self.scene['robot'].data.root_pos_w
        self.robot_root_quat = self.scene['robot'].data.root_quat_w

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = slice(None)

        # Support both left and right arm based on eef_name
        # eef_name can be: "left_arm", "right_arm", or robot name (defaults to right arm)
        if "right" in eef_name.lower():
            # Default to right arm (primary manipulator for Mimic tracking)
            eef_pose = self.obs_buf["policy"]["right_eef_pose"][env_ids]
        elif "left" in eef_name.lower():
            eef_pose = self.obs_buf["policy"]["left_eef_pose"][env_ids]
        else:
            print("Defaulting to right arm EEF state.")
            eef_pose = self.obs_buf["policy"]["right_eef_pose"][env_ids]

        eef_pos = eef_pose[:, :3]
        eef_quat = eef_pose[:, 3:7]
        # quat: (w, x, y, z)
        eef_pose = PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))

        return eef_pose

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        # Mimic framework provides single EEF pose (right arm as primary)
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        if "right" in eef_name.lower():
            target_right_eef_pose = target_eef_pose_dict[eef_name]
            right_gripper_action = gripper_action_dict[eef_name]

            # Convert right EEF pose to pos + quat
            right_eef_pos, right_eef_rot = PoseUtils.unmake_pose(target_right_eef_pose)
            right_eef_quat = PoseUtils.quat_from_matrix(right_eef_rot)

            right_pose_action = torch.cat([right_eef_pos, right_eef_quat], dim=0)

            # For left arm, keep current observation state (not controlled by Mimic trajectory)
            # Get current left arm EEF state from observation
            left_eef_pose = self.obs_buf["policy"]["left_eef_pose"]

            if left_eef_pose.dim() > 1:
                left_eef_pose = left_eef_pose[env_id]
            left_pose_action = left_eef_pose[:7]  # pos(3) + quat(4)

            # For gripper_l, keep current state
            joint_pos_target = self.obs_buf["policy"]["joint_pos_target"]

            if joint_pos_target.dim() > 1:
                left_gripper_action = joint_pos_target[env_id, 7:8]
                head_action = joint_pos_target[env_id, 16:18]
                lift_action = joint_pos_target[env_id, 18:19]
            else:
                left_gripper_action = joint_pos_target[7:8]
                lift_action = joint_pos_target[18:19]
                head_action = joint_pos_target[16:18]

            # Concatenate full 20D action:
            # [left_eef(7), gripper_l(1), right_eef(7), gripper_r(1), lift(1), head(2)]
            action = torch.cat([
                left_pose_action,     # 1-7: left arm (keep current)
                left_gripper_action,  # 8: left gripper (keep current)
                right_pose_action,      # 9-15: right arm (Mimic controlled)
                right_gripper_action,      # 16: right gripper (keep current)
                head_action,           # 17-18: head (keep current)
                lift_action            # 19: lift (keep current)
            ], dim=0)

            result = action.unsqueeze(0)
        elif "left" in eef_name.lower():
            target_left_eef_pose = target_eef_pose_dict[eef_name]
            left_gripper_action = gripper_action_dict[eef_name]

            # Convert left EEF pose to pos + quat
            left_eef_pos, left_eef_rot = PoseUtils.unmake_pose(target_left_eef_pose)
            left_eef_quat = PoseUtils.quat_from_matrix(left_eef_rot)

            left_pose_action = torch.cat([left_eef_pos, left_eef_quat], dim=0)

            # For right arm, keep current observation state (not controlled by Mimic trajectory)
            # Get current right arm EEF state from observation
            right_eef_pose = self.obs_buf["policy"]["right_eef_pose"]

            if right_eef_pose.dim() > 1:
                right_eef_pose = right_eef_pose[env_id]
            right_pose_action = right_eef_pose[:7]  # pos(3) + quat(4)

            # For gripper_l, keep current state
            joint_pos_target = self.obs_buf["policy"]["joint_pos_target"]

            if joint_pos_target.dim() > 1:
                right_gripper_action = joint_pos_target[env_id, 15:16]
                head_action = joint_pos_target[env_id, 16:18]
                lift_action = joint_pos_target[env_id, 18:19]
            else:
                right_gripper_action = joint_pos_target[15:16]
                lift_action = joint_pos_target[18:19]
                head_action = joint_pos_target[16:18]

            # Concatenate full 20D action:
            # [left_eef(7), gripper_l(1), right_eef(7), gripper_r(1), lift(1), head(2)]
            action = torch.cat([
                left_pose_action,     # 1-7: left arm (Mimic controlled)
                left_gripper_action,  # 8: left gripper (Mimic controlled)
                right_pose_action,      # 9-15: right arm (keep current)
                right_gripper_action,      # 16: right gripper (keep current)
                lift_action,           # 17: lift (keep current)
                head_action            # 18-19: head (keep current)
            ], dim=0)

            result = action.unsqueeze(0)
        return result

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        # For FFW-SG2, use right arm as primary manipulator
        # Action format from IK conversion: [left_eef(7), gripper_l(1), right_eef(7), gripper_r(1), lift(1), head(2)]
        # We return only the right arm EEF pose (indices 8-14)
        if "right" in eef_name.lower():
            target_eef_pos = action[:, 8:11]    # Right arm position
            target_eef_quat = action[:, 11:15]  # Right arm quaternion
            target_eef_rot = PoseUtils.matrix_from_quat(target_eef_quat)
        elif "left" in eef_name.lower():
            target_eef_pos = action[:, 0:3]    # Left arm position
            target_eef_quat = action[:, 3:7]  # Left arm quaternion
            target_eef_rot = PoseUtils.matrix_from_quat(target_eef_quat)
        else:
            print("Defaulting to right arm EEF state.")
            target_eef_pos = action[:, 8:11]    # Right arm position
            target_eef_quat = action[:, 11:15]  # Right arm quaternion
            target_eef_rot = PoseUtils.matrix_from_quat(target_eef_quat)

        target_eef_pose = PoseUtils.make_pose(target_eef_pos, target_eef_rot).clone()

        return {eef_name: target_eef_pose}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        # For FFW-SG2, return right gripper (index 15)
        # Action format: [left_eef(7), gripper_l(1), right_eef(7), gripper_r(1), lift(1), head(2)]
        if "right" in eef_name.lower():
            target_gripper_action = actions[:, 15:16]
        elif "left" in eef_name.lower():
            target_gripper_action = actions[:, 7:8]
        else:
            print("Defaulting to right gripper action.")
            target_gripper_action = actions[:, 15:16]
        return {eef_name: target_gripper_action}

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """
        Gets a dictionary of termination signal flags for each subtask in a task. The flag is 1
        when the subtask has been completed and 0 otherwise. The implementation of this method is
        required if intending to enable automatic subtask term signal annotation when running the
        dataset annotation tool. This method can be kept unimplemented if intending to use manual
        subtask term signal annotation.

        Args:
            env_ids: Environment indices to get the termination signals for. If None, all envs are considered.

        Returns:
            A dictionary termination signal flags (False or True) for each subtask.
        """
        if env_ids is None:
            env_ids = slice(None)

        signals = dict()
        subtask_terms = self.obs_buf["subtask_terms"]
        for term_name, term_signal in subtask_terms.items():
            signals[term_name] = term_signal[env_ids]

        return signals
