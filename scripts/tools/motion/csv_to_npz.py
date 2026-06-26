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
# Additional notices:
# This module is adapted from HybridRobotics/whole_body_tracking, licensed under the MIT License.
# See THIRD_PARTY_LICENSES.md for details.

"""Replay a K1 Rev.1 23-DoF motion CSV and export converted CSV plus NPZ.

The input CSV is K1 Rev.1-native: root pose (3 position + 4 quaternion columns)
followed by exactly 23 K1 Rev.1 joint columns.

Example:
    ./third_party/IsaacLab/isaaclab.sh -p scripts/tools/motion/csv_to_npz.py \
        -f /path/to/dance1_raw.csv \
        --output_name /path/to/dance1.npz \
        --input_fps 50 --output_fps 50 --root-quat-order xyzw --headless

This writes:
    /path/to/dance1.csv  (converted/resampled CSV)
    /path/to/dance1.npz

If ``--output_name`` is omitted, the exported CSV and NPZ use the
``<input_stem>_converted`` stem so the source CSV is preserved.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

import numpy as np

from isaaclab.app import AppLauncher


K1_REV1_MOTION_CSV_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
]


parser = argparse.ArgumentParser(description="Replay K1 Rev.1 motion CSV and export converted CSV plus NPZ.")
parser.add_argument("--input_file", "-f", type=str, required=True, help="Input K1 Rev.1 motion CSV.")
parser.add_argument("--input_fps", type=int, default=50, help="FPS of the input CSV.")
parser.add_argument("--output_fps", type=int, default=50, help="FPS of the exported CSV/NPZ.")
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help="1-based inclusive input frame range. If omitted, all frames are used.",
)
parser.add_argument(
    "--output_name",
    type=str,
    help="Output NPZ path. The converted CSV is written next to it with the same stem.",
)
parser.add_argument(
    "--root-quat-order",
    type=str,
    default="xyzw",
    choices=("xyzw", "wxyz"),
    help="Order of the 4 root quaternion columns in the input CSV. Output CSV is always xyzw.",
)
parser.add_argument(
    "--smooth-window",
    type=int,
    default=1,
    help="Optional odd moving-average window in input frames. Values <= 1 disable smoothing.",
)
parser.add_argument("--smooth-passes", type=int, default=1, help="Number of smoothing passes.")
parser.add_argument(
    "--smooth-fields",
    nargs="+",
    default=("root_pos", "root_rot", "joints"),
    choices=("root_pos", "root_rot", "joints", "all"),
    help="Motion channels to smooth.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if not args_cli.output_name:
    input_root, _ = os.path.splitext(args_cli.input_file)
    args_cli.output_name = input_root + "_converted.npz"
else:
    output_root, output_ext = os.path.splitext(args_cli.output_name)
    if output_ext.lower() == ".csv":
        args_cli.output_name = output_root + ".npz"
    elif output_ext.lower() != ".npz":
        args_cli.output_name = args_cli.output_name + ".npz"


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import torch.nn.functional as F

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

from cyclo_lab.assets.robots.K1_rev1 import K1_REV1_INERTIA_TUNED_CFG


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Scene used to replay K1 Rev.1 23-DoF motion."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = K1_REV1_INERTIA_TUNED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class MotionLoader:
    """Loads, optionally smooths, resamples, and differentiates K1 Rev.1 23-DoF CSV motion."""

    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,
        root_quat_order: str,
        smooth_window: int,
        smooth_passes: int,
        smooth_fields: tuple[str, ...],
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / input_fps
        self.output_dt = 1.0 / output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self.root_quat_order = root_quat_order
        self.smooth_window = smooth_window
        self.smooth_passes = smooth_passes
        self.smooth_fields = self._normalize_smooth_fields(smooth_fields)
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        if self.frame_range is None:
            motion_np = np.loadtxt(self.motion_file, delimiter=",")
        else:
            motion_np = np.loadtxt(
                self.motion_file,
                delimiter=",",
                skiprows=self.frame_range[0] - 1,
                max_rows=self.frame_range[1] - self.frame_range[0] + 1,
            )
        if motion_np.ndim == 1:
            motion_np = motion_np[None, :]
        if motion_np.shape[1] != 30:
            raise ValueError(
                "K1 Rev.1 23-DoF CSV must have 30 columns "
                f"(root 7 + 23 joints), got {motion_np.shape[1]}: {self.motion_file}"
            )

        motion = torch.from_numpy(motion_np).to(torch.float32).to(self.device)
        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = motion[:, 3:7]
        if self.root_quat_order == "xyzw":
            self.motion_base_rots_input = self.motion_base_rots_input[:, [3, 0, 1, 2]]
        self.motion_base_rots_input = self._normalize_quaternions(self.motion_base_rots_input)
        self.motion_dof_poss_input = motion[:, 7:]
        self.input_frames = motion.shape[0]
        if self.input_frames < 2:
            raise ValueError("Motion must contain at least 2 frames.")
        self._smooth_motion_input()
        self.duration = (self.input_frames - 1) * self.input_dt
        print(f"Motion loaded ({self.motion_file}), duration: {self.duration} sec, frames: {self.input_frames}")

    def _normalize_smooth_fields(self, fields: tuple[str, ...]) -> set[str]:
        selected = set(fields)
        if "all" in selected:
            return {"root_pos", "root_rot", "joints"}
        return selected

    def _smooth_motion_input(self):
        if self.smooth_window <= 1:
            return
        if self.smooth_passes < 1:
            raise ValueError(f"--smooth-passes must be >= 1, got {self.smooth_passes}.")
        if self.smooth_window % 2 == 0:
            self.smooth_window += 1
            print(f"[INFO] --smooth-window must be odd; using {self.smooth_window}.")

        for _ in range(self.smooth_passes):
            if "root_pos" in self.smooth_fields:
                self.motion_base_poss_input = self._moving_average(self.motion_base_poss_input, self.smooth_window)
            if "root_rot" in self.smooth_fields:
                self.motion_base_rots_input = self._smooth_quaternions(self.motion_base_rots_input, self.smooth_window)
            if "joints" in self.smooth_fields:
                self.motion_dof_poss_input = self._moving_average(self.motion_dof_poss_input, self.smooth_window)

        print(
            "[INFO] Applied motion smoothing: "
            f"window={self.smooth_window}, passes={self.smooth_passes}, fields={sorted(self.smooth_fields)}"
        )

    def _moving_average(self, values: torch.Tensor, window: int) -> torch.Tensor:
        padding = window // 2
        channels = values.shape[1]
        x = values.transpose(0, 1).unsqueeze(0)
        x = F.pad(x, (padding, padding), mode="replicate")
        kernel = torch.full((channels, 1, window), 1.0 / window, dtype=values.dtype, device=values.device)
        smoothed = F.conv1d(x, kernel, groups=channels)
        return smoothed.squeeze(0).transpose(0, 1)

    def _smooth_quaternions(self, quats_wxyz: torch.Tensor, window: int) -> torch.Tensor:
        quats = self._normalize_quaternions(quats_wxyz)
        quats = self._make_quaternion_sign_continuous(quats)
        quats = self._moving_average(quats, window)
        return self._normalize_quaternions(quats)

    def _normalize_quaternions(self, quats_wxyz: torch.Tensor) -> torch.Tensor:
        return quats_wxyz / torch.clamp(torch.linalg.norm(quats_wxyz, dim=1, keepdim=True), min=1.0e-8)

    def _make_quaternion_sign_continuous(self, quats_wxyz: torch.Tensor) -> torch.Tensor:
        out = quats_wxyz.clone()
        for idx in range(1, out.shape[0]):
            if torch.dot(out[idx - 1], out[idx]) < 0.0:
                out[idx] = -out[idx]
        return out

    def _interpolate_motion(self):
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0], self.motion_base_poss_input[index_1], blend.unsqueeze(1)
        )
        self.motion_base_rots = self._slerp(self.motion_base_rots_input[index_0], self.motion_base_rots_input[index_1], blend)
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0], self.motion_dof_poss_input[index_1], blend.unsqueeze(1)
        )
        print(
            f"Motion interpolated, input frames: {self.input_frames}, input fps: {self.input_fps}, "
            f"output frames: {self.output_frames}, output fps: {self.output_fps}"
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor):
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1, device=self.device))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        return torch.cat([omega[:1], omega, omega[-1:]], dim=0)

    def get_next_state(self):
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


def _get_output_csv_path(output_name: str) -> str:
    output_stem, _ = os.path.splitext(output_name)
    return output_stem + ".csv"


def run_simulator(sim: SimulationContext, scene: InteractiveScene):
    motion = MotionLoader(
        motion_file=args_cli.input_file,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=args_cli.frame_range,
        root_quat_order=args_cli.root_quat_order,
        smooth_window=args_cli.smooth_window,
        smooth_passes=args_cli.smooth_passes,
        smooth_fields=tuple(args_cli.smooth_fields),
    )

    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(K1_REV1_MOTION_CSV_JOINT_NAMES, preserve_order=True)[0]
    if len(robot_joint_indexes) != len(K1_REV1_MOTION_CSV_JOINT_NAMES):
        raise RuntimeError("Could not resolve all K1 Rev.1 23-DoF CSV joints in the robot asset.")

    motion_base_rot_xyzw = motion.motion_base_rots[:, [1, 2, 3, 0]]
    motion_csv = torch.cat((motion.motion_base_poss, motion_base_rot_xyzw, motion.motion_dof_poss), dim=1)
    output_csv_path = _get_output_csv_path(args_cli.output_name)
    np.savetxt(output_csv_path, motion_csv.cpu().numpy(), delimiter=",", fmt="%.9f")
    print("[INFO]: Converted K1 Rev.1 23-DoF motion csv file saved to", output_csv_path)
    print("[INFO]: CSV joint order:")
    for idx, joint_name in enumerate(K1_REV1_MOTION_CSV_JOINT_NAMES):
        print(f"  {idx:02d}: {joint_name}")

    log = {
        "fps": [args_cli.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    for _ in range(motion.output_frames):
        (
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ),
            reset_flag,
        ) = motion.get_next_state()

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.render()
        scene.update(sim.get_physics_dt())

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
        log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

        if reset_flag:
            break

    for key in (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    ):
        log[key] = np.stack(log[key], axis=0)
    np.savez(args_cli.output_name, **log)
    print("[INFO]: Motion npz file saved to", args_cli.output_name)
    print("[INFO]: Conversion complete; closing simulator.")


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    os._exit(0)
