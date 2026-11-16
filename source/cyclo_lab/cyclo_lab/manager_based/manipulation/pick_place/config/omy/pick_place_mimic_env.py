# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from collections.abc import Sequence

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv, ManagerBasedRLEnvCfg


class OMYPickPlaceMimicEnv(ManagerBasedRLMimicEnv):
    """
    Isaac Lab Mimic environment wrapper class for OMY Cube Stack IK Rel env.
    """

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot_root_pos = self.scene['robot'].data.root_pos_w.torch
        self.robot_root_quat = self.scene['robot'].data.root_quat_w.torch

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = slice(None)

        # robot coordinate
        eef_pose = self.obs_buf["policy"]["eef_pose"][env_ids]
        eef_pos = eef_pose[:, :3]
        eef_quat = eef_pose[:, 3:7]
        # quat: (x, y, z, w)
        eef_pose = PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))

        # print("eef_pose: ", eef_pose)

        return eef_pose

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        (target_eef_pose,) = target_eef_pose_dict.values()
        target_eef_pos, target_eef_rot = PoseUtils.unmake_pose(target_eef_pose)
        target_eef_quat = PoseUtils.quat_from_matrix(target_eef_rot)

        (gripper_action,) = gripper_action_dict.values()

        eef_name = next(iter(self.cfg.subtask_configs))
        if action_noise_dict is not None and action_noise_dict[eef_name] != 0:
            # Perturb translation and axis-angle rotation, preserving a unit quaternion.
            delta_pose = target_eef_pos.new_empty((1, 6)).normal_() * action_noise_dict[eef_name]
            target_eef_pos, target_eef_quat = PoseUtils.apply_delta_pose(
                target_eef_pos.unsqueeze(0), target_eef_quat.unsqueeze(0), delta_pose
            )
            target_eef_pos, target_eef_quat = target_eef_pos[0], target_eef_quat[0]
        pose_action = torch.cat([target_eef_pos, target_eef_quat], dim=0)

        return torch.cat([pose_action, gripper_action], dim=0).unsqueeze(0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        target_eef_pos = action[:, :3]
        target_eef_quat = action[:, 3:7]
        target_eef_rot = PoseUtils.matrix_from_quat(target_eef_quat)

        target_eef_pose = PoseUtils.make_pose(target_eef_pos, target_eef_rot).clone()

        # print("target_eef_pose: ", target_eef_pose)

        return {eef_name: target_eef_pose}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        eef_name = list(self.cfg.subtask_configs.keys())[0]
        return {eef_name: actions[:, -1:]}

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
