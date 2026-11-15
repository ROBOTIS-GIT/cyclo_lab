"""Tests for TaskSpec velocity extraction and registry resolution."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import torch

from supermimic.tasks.task_spec import (
  TasksCfg,
  make_velocity_task,
  resolve_task_specs,
)
from tests.conftest import identity_aux


class FakeMotion:
  """Minimal MotionLoader stand-in: just needs root_lin_vel_w + root_ang_vel_w."""

  def __init__(
    self, root_lin_vel_w: torch.Tensor, root_ang_vel_w: torch.Tensor
  ) -> None:
    self.root_lin_vel_w = root_lin_vel_w
    self.root_ang_vel_w = root_ang_vel_w


def _velocity_task(
  window_frames: int = 0,
  command_noise: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
  """make_velocity_task with permissive ranges so projection/noise tests
  exercise the math without hitting cfg clamps."""
  return make_velocity_task(
    lin_vel_x_range=(-10.0, 10.0),
    lin_vel_y_range=(-10.0, 10.0),
    ang_vel_z_range=(-10.0, 10.0),
    window_frames=window_frames,
    command_noise=command_noise,
  )


class TestVelocityExtraction:
  def test_body_frame_passthrough_when_robot_aligned_with_world(self):
    """Identity quaternion → body frame == world frame; cmd = [vx_w, vy_w, wz_w]."""
    lin = torch.zeros(5, 3)
    ang = torch.zeros(5, 3)
    lin[2] = torch.tensor([1.0, 0.5, 0.7])  # z-component should be dropped
    ang[2] = torch.tensor([0.2, 0.3, 1.5])  # only z used for yaw rate
    motion = FakeMotion(lin, ang)

    velocity_task = _velocity_task()
    motion_times = torch.tensor([2])
    aux = identity_aux(1)
    goal = velocity_task.extract_training_goal(motion, motion_times, aux)
    assert goal.shape == (1, 3)
    assert torch.allclose(goal[0], torch.tensor([1.0, 0.5, 1.5]), atol=1e-5)

  def test_yawed_robot_projects_into_body_frame(self):
    """Robot facing +y, motion lin vel in +x → body-frame x = 0, body-frame y = -1."""
    lin = torch.zeros(5, 3)
    ang = torch.zeros(5, 3)
    lin[0] = torch.tensor([1.0, 0.0, 0.0])
    ang[0] = torch.tensor([0.0, 0.0, 0.8])  # world yaw rate = body yaw rate
    motion = FakeMotion(lin, ang)

    velocity_task = _velocity_task()
    aux = torch.zeros(1, 7)
    # yaw = +π/2 about z: quat (w, x, y, z) = (cos π/4, 0, 0, sin π/4).
    aux[0, 3] = math.cos(math.pi / 4)
    aux[0, 6] = math.sin(math.pi / 4)
    motion_times = torch.tensor([0])
    goal = velocity_task.extract_training_goal(motion, motion_times, aux)
    # World +x rotated into body frame of a +π/2-yawed robot → body -y.
    assert torch.allclose(goal[0], torch.tensor([0.0, -1.0, 0.8]), atol=1e-5)


class TestCenteredWindow:
  def test_window_averages_centered_motion_velocity(self):
    """window_frames=3 at time 4 averages frames {3, 4, 5} of root_lin_vel_w."""
    lin = torch.zeros(10, 3)
    ang = torch.zeros(10, 3)
    lin[3] = torch.tensor([1.0, 0.0, 0.0])
    lin[4] = torch.tensor([2.0, 0.0, 0.0])
    lin[5] = torch.tensor([3.0, 0.0, 0.0])
    motion = FakeMotion(lin, ang)

    task = _velocity_task(window_frames=3)
    goal = task.extract_training_goal(motion, torch.tensor([4]), identity_aux(1))
    # mean of {1, 2, 3} = 2 on the x-axis.
    assert torch.allclose(goal[0, 0], torch.tensor(2.0), atol=1e-5)

  def test_window_clamps_at_clip_start(self):
    """Indices before t=0 clamp to frame 0."""
    lin = torch.zeros(5, 3)
    ang = torch.zeros(5, 3)
    # window=3 at t=0 → indices {-1, 0, 1} → clamped {0, 0, 1}.
    # lin[0]=3, lin[1]=0 → mean = (3+3+0)/3 = 2.0.
    lin[0] = torch.tensor([3.0, 0.0, 0.0])
    motion = FakeMotion(lin, ang)

    task = _velocity_task(window_frames=3)
    goal = task.extract_training_goal(motion, torch.tensor([0]), identity_aux(1))
    assert torch.allclose(goal[0, 0], torch.tensor(2.0), atol=1e-5)

  def test_window_clamps_at_clip_end(self):
    """Indices past the last frame clamp to the last frame."""
    lin = torch.zeros(5, 3)
    ang = torch.zeros(5, 3)
    # 5 frames (0..4). window=3 at t=4 → indices {3, 4, 5} → clamped {3, 4, 4}.
    # lin[3]=0, lin[4]=3 → mean = (0+3+3)/3 = 2.0.
    lin[4] = torch.tensor([3.0, 0.0, 0.0])
    motion = FakeMotion(lin, ang)

    task = _velocity_task(window_frames=3)
    goal = task.extract_training_goal(motion, torch.tensor([4]), identity_aux(1))
    assert torch.allclose(goal[0, 0], torch.tensor(2.0), atol=1e-5)


class TestCommandNoise:
  def test_zero_motion_with_noise_stays_in_half_width(self):
    """With reference = 0 and noise half-width h, output is in [-h, +h]."""
    motion = FakeMotion(torch.zeros(2, 3), torch.zeros(2, 3))
    task = _velocity_task(command_noise=(0.4, 0.2, 0.6))
    motion_times = torch.zeros(64, dtype=torch.long)
    cmd = task.extract_training_goal(motion, motion_times, identity_aux(64))
    assert cmd[:, 0].min() >= -0.4 - 1e-6 and cmd[:, 0].max() <= 0.4 + 1e-6
    assert cmd[:, 1].min() >= -0.2 - 1e-6 and cmd[:, 1].max() <= 0.2 + 1e-6
    assert cmd[:, 2].min() >= -0.6 - 1e-6 and cmd[:, 2].max() <= 0.6 + 1e-6

  def test_clamped_to_cfg_ranges(self):
    """Huge noise still bounded by configured ranges (cfg lo/hi clamp)."""
    motion = FakeMotion(torch.zeros(2, 3), torch.zeros(2, 3))
    task = make_velocity_task(
      lin_vel_x_range=(-1.0, 1.0),
      lin_vel_y_range=(-0.5, 0.5),
      ang_vel_z_range=(-1.0, 1.0),
      window_frames=0,
      command_noise=(100.0, 100.0, 100.0),
    )
    motion_times = torch.zeros(64, dtype=torch.long)
    cmd = task.extract_training_goal(motion, motion_times, identity_aux(64))
    assert cmd[:, 0].min() >= -1.0 and cmd[:, 0].max() <= 1.0
    assert cmd[:, 1].min() >= -0.5 and cmd[:, 1].max() <= 0.5
    assert cmd[:, 2].min() >= -1.0 and cmd[:, 2].max() <= 1.0


class TestInferenceHelpers:
  def test_build_inference_goal_obs_is_identity(self):
    """Raw goal is already body-frame; build_obs is a passthrough."""
    velocity_task = _velocity_task()
    raw = torch.tensor([[0.5, -0.2, 0.3]])
    obs = velocity_task.build_inference_goal_obs(raw)
    assert obs.shape == (1, 3)
    assert torch.allclose(obs[0], raw[0])


class TestResolveTaskSpecs:
  def test_motion_and_velocity(self):
    rep = MagicMock()
    specs = resolve_task_specs(TasksCfg(), rep, num_goals=87)
    names = [s.name for s in specs]
    assert names == ["motion", "velocity"]
    # motion goal_dim = num_goals - rel_pos_dim (3 * NUM_GOAL_FRAMES = 30) = 57
    assert specs[0].goal_dim == 57
    assert specs[1].goal_dim == 3
    assert specs[1].raw_goal_dim == 3
