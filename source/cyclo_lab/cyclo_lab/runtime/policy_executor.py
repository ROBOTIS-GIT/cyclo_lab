#!/usr/bin/env python3

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

import os

import numpy as np
import torch
import yaml


class PolicyExecutor:
    def __init__(self):
        self.policy: torch.jit.ScriptModule | None = None
        self.yaml_data: dict | None = None

    def load_policy_yaml(self, policy_yaml_path: str) -> None:
        if not os.path.isfile(policy_yaml_path):
            raise FileNotFoundError(f"Policy YAML file not found or is a directory: {policy_yaml_path}")
        with open(policy_yaml_path, "r") as file:
            # WARNING: UnsafeLoader can execute arbitrary code and is a security risk.
            # Only use with trusted YAML files. Prefer SafeLoader if possible.
            data = yaml.load(file, Loader=yaml.UnsafeLoader)
            self.yaml_data = data

    def get_yaml_data(self, key_path: str, default=None):
        if self.yaml_data is None:
            raise ValueError("YAML not loaded. Call `load_policy_yaml()` first.")
        keys = key_path.split(".")
        val = self.yaml_data
        try:
            for key in keys:
                val = val[key]
            return val
        except (KeyError, TypeError):
            return default

    def load_policy_model(self, policy_model_path: str) -> None:
        if not os.path.exists(policy_model_path):
            raise FileNotFoundError(f"Policy model file not found: {policy_model_path}")
        self.policy = torch.jit.load(policy_model_path)

    def update_action(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs = torch.from_numpy(obs).view(1, -1).float()
            action = self.policy(obs).detach().view(-1).numpy()
        return action

    def get_action_scale(self) -> float:
        """Get the action scale from the YAML configuration."""
        if self.yaml_data is None:
            raise ValueError("YAML not loaded. Call `load_policy_yaml()` first.")
        return self.get_yaml_data("actions.arm_action.scale", 0.5)

    def get_default_joint_positions(self, joint_names: list) -> np.ndarray:
        """Get the default joint positions from the YAML configuration for given joint names."""
        if self.yaml_data is None:
            raise ValueError("YAML not loaded. Call `load_policy_yaml()` first.")

        default_joint_yaml = self.get_yaml_data("scene.robot.init_state.joint_pos", None)
        if default_joint_yaml is None:
            raise ValueError("Default joint positions not found in YAML.")

        try:
            joint_pos_array = np.array([float(default_joint_yaml[name]) for name in joint_names], dtype=np.float32)
        except KeyError as e:
            raise KeyError(f"Joint name {e} not found in YAML joint_pos dict.")

        return joint_pos_array

    def get_observation_joint_names(self) -> list:
        """Get the observation joint names from the YAML configuration."""
        if self.yaml_data is None:
            raise ValueError("YAML not loaded. Call `load_policy_yaml()` first.")
        return self.get_yaml_data("observations.policy.joint_pos.params.asset_cfg.joint_names", [])
