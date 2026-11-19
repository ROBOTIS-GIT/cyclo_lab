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
# Author: Kiwoong Park

"""Deployment configuration export for BlinkMimic."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ObservationField:
    name: str
    width: int
    scale: float = 1.0


def observation_fields(num_joints):
    return (
        ObservationField("joint_pos_rel", num_joints),
        ObservationField("joint_vel", num_joints),
        ObservationField("base_ang_vel", 3),
        ObservationField("projected_gravity", 3),
        ObservationField("last_action", num_joints),
        ObservationField("reference_orientation", 6),
    )


def goal_fields(num_joints):
    return (
        ObservationField("future_root_displacement", 3),
        ObservationField("future_root_orientation", 6),
        ObservationField("future_root_linear_velocity", 3),
        ObservationField("future_root_angular_velocity", 3),
        ObservationField("future_joint_pos", num_joints),
        ObservationField("future_joint_vel", num_joints, 0.1),
    )


def native_dimensions(num_joints):
    from .mdp import obs_dim
    from .motion import GOAL_Q_OFFSET

    return obs_dim(num_joints), GOAL_Q_OFFSET + 2 * num_joints


def validate_field_dimensions(num_joints):
    """Fail export when the upstream layout no longer fits the named fields."""
    expected = native_dimensions(num_joints)
    actual = tuple(
        sum(field.width for field in fields)
        for fields in (
            observation_fields(num_joints),
            goal_fields(num_joints),
        )
    )
    if actual != expected:
        raise ValueError(f"BlinkMimic field dimensions {actual} differ from upstream {expected}")


def observation_config(history, future, num_joints):
    """Describe the native actor input before ONNX normalization."""
    validate_field_dimensions(num_joints)

    def terms(fields, history_length=None):
        return {
            field.name: dict(
                scale=[field.scale] * field.width,
                clip=None,
                **({"history_length": history_length} if history_length else {}),
            )
            for field in fields
        }

    sensor = terms(observation_fields(num_joints), history)
    goal = terms(goal_fields(num_joints))
    return dict(
        history=dict(layout="frame_major", terms=sensor),
        motion_goal=dict(
            frames=future, stride=1, start_offset=1, groups=[dict(layout="frame_major", terms=goal)]
        ),
    )
