"""The physical pose and its reference must belong to the same policy tick."""

from types import SimpleNamespace

import pytest
import torch

from cyclo_lab.manager_based.mimic._shared.motion_library_command import MotionLibraryCommand
from cyclo_lab.manager_based.mimic.beyondmimic.mdp.terminations import (
    reference_anchor_gravity_deviation,
    reference_anchor_height_deviation,
)


@pytest.mark.parametrize("targets_next_frame", [False, True], ids=["blink", "super"])
def test_post_step_termination_uses_correct_frame_and_stays_inside_clip(targets_next_frame):
    class Command:
        termination_frame_ids = MotionLibraryCommand.termination_frame_ids
        anchor_pos_w_at = MotionLibraryCommand.anchor_pos_w_at
        anchor_quat_w_at = MotionLibraryCommand.anchor_quat_w_at

    command = Command()
    command.cfg = SimpleNamespace(clip_timeout=True, targets_next_frame=targets_next_frame)
    command.frame_ids = torch.arange(4)
    command.motion_ends = torch.tensor([2, 4])
    command.reference_anchor_body_id = 0
    pos = torch.zeros(4, 1, 3)
    pos[1::2, 0, 2] = 1
    quat = torch.zeros(4, 1, 4)
    quat[::2, 0, 3] = 1
    quat[1::2, 0, 0] = 1
    command.reference = SimpleNamespace(body_position_w=pos, body_orientation_w=quat)
    command._env = SimpleNamespace(scene=SimpleNamespace(env_origins=torch.zeros(4, 3)))
    expected = torch.tensor([0, 1, 2, 3] if targets_next_frame else [1, 1, 3, 3])
    command.robot_anchor_pos_w = pos[expected, 0]
    command.robot_anchor_quat_w = quat[expected, 0]

    gravity = SimpleNamespace(torch=torch.tensor([[0.0, 0.0, -1.0]]).expand(4, -1))
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda name: command),
        scene={"robot": SimpleNamespace(data=SimpleNamespace(GRAVITY_VEC_W=gravity))},
    )
    assert torch.equal(command.termination_frame_ids, expected)
    assert not reference_anchor_height_deviation(env, "motion", 0.25).any()
    assert not reference_anchor_gravity_deviation(env, SimpleNamespace(name="robot"), "motion", 0.8).any()
    command.robot_anchor_pos_w = command.robot_anchor_pos_w + 1
    command.robot_anchor_quat_w = command.robot_anchor_quat_w[:, [3, 1, 2, 0]]
    assert reference_anchor_height_deviation(env, "motion", 0.25).all()
    assert reference_anchor_gravity_deviation(env, SimpleNamespace(name="robot"), "motion", 0.8).all()
