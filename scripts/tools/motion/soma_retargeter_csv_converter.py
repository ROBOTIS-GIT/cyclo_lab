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

"""Convert Soma retargeter CSV files to K1 Rev.1 motion CSV and NPZ.

The Soma retargeter CSV format is headered and contains:

* Frame
* root_translateX, root_translateY, root_translateZ
* root_rotateX, root_rotateY, root_rotateZ
* 23 K1 Rev.1 joint columns named ``<joint_name>_dof``

By default, root translation is interpreted as centimeters and all rotations as
degrees. The default FPS conversion is 30 Hz input to 50 Hz output.

Example:
    ./third_party/IsaacLab/isaaclab.sh -p scripts/tools/motion/soma_retargeter_csv_converter.py \
        -f source/cyclo_lab/data/motions/K1_rev1/dance1/dance1_soma.csv \
        --output_name source/cyclo_lab/data/motions/K1_rev1/dance1/dance1.npz \
        --headless

This writes:
    source/cyclo_lab/data/motions/K1_rev1/dance1/dance1.csv
    source/cyclo_lab/data/motions/K1_rev1/dance1/dance1.npz

If ``--output_name`` is omitted, the exported CSV and NPZ use the
``<input_stem>_converted`` stem so the source CSV is preserved.

The exported motion CSV is numeric only:

* root position in meters
* root quaternion in xyzw order
* 23 joint positions in radians

The exported NPZ has the same keys expected by the mimic reference dataloader:
``fps``, ``joint_pos``, ``joint_vel``, ``body_pos_w``, ``body_quat_w``,
``body_lin_vel_w``, and ``body_ang_vel_w``.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
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

SOMA_ROOT_POSITION_COLUMNS = ["root_translateX", "root_translateY", "root_translateZ"]
SOMA_ROOT_EULER_COLUMNS = ["root_rotateX", "root_rotateY", "root_rotateZ"]
SOMA_JOINT_COLUMNS = [f"{name}_dof" for name in K1_REV1_MOTION_CSV_JOINT_NAMES]
SOMA_REQUIRED_COLUMNS = ["Frame", *SOMA_ROOT_POSITION_COLUMNS, *SOMA_ROOT_EULER_COLUMNS, *SOMA_JOINT_COLUMNS]


parser = argparse.ArgumentParser(description="Convert Soma retargeter CSV to K1 Rev.1 motion CSV plus NPZ.")
parser.add_argument("--input_file", "-f", type=str, required=True, help="Input Soma retargeter CSV.")
parser.add_argument("--input_fps", type=int, default=30, help="FPS of the input CSV.")
parser.add_argument("--output_fps", type=int, default=50, help="FPS of the exported CSV/NPZ.")
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help="1-based inclusive input row range after the header. If omitted, all frames are used.",
)
parser.add_argument(
    "--output_name",
    type=str,
    help=(
        "Output NPZ path. The motion CSV is written next to it with the same stem. "
        "If omitted, '<input_stem>_converted.npz' is used to preserve the input CSV."
    ),
)
parser.add_argument(
    "--position-scale",
    type=float,
    default=0.01,
    help="Scale applied to root translations before export. Default converts centimeters to meters.",
)
parser.add_argument(
    "--angle-unit",
    type=str,
    default="degrees",
    choices=("degrees", "radians"),
    help="Unit for root Euler angles and joint columns in the input CSV.",
)
parser.add_argument(
    "--root-height-offset",
    type=float,
    default=0.0,
    help="Additional z offset in meters after applying --position-scale.",
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
from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_conjugate,
    quat_from_euler_xyz,
    quat_mul,
    quat_slerp,
)

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
    """Loads, converts, resamples, and differentiates Soma retargeter CSV motion."""

    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,
        position_scale: float,
        angle_unit: str,
        root_height_offset: float,
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
        self.position_scale = position_scale
        self.angle_unit = angle_unit
        self.root_height_offset = root_height_offset
        self.smooth_window = smooth_window
        self.smooth_passes = smooth_passes
        self.smooth_fields = self._normalize_smooth_fields(smooth_fields)
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        motion_np = self._read_soma_csv()
        motion = torch.from_numpy(motion_np).to(torch.float32).to(self.device)

        root_pos = motion[:, :3] * self.position_scale
        root_pos[:, 2] += self.root_height_offset

        root_euler = motion[:, 3:6]
        joint_pos = motion[:, 6:]
        if self.angle_unit == "degrees":
            root_euler = torch.deg2rad(root_euler)
            joint_pos = torch.deg2rad(joint_pos)

        self.motion_base_poss_input = root_pos
        self.motion_base_rots_input = quat_from_euler_xyz(root_euler[:, 0], root_euler[:, 1], root_euler[:, 2])
        self.motion_base_rots_input = self._normalize_quaternions(self.motion_base_rots_input)
        self.motion_dof_poss_input = joint_pos
        self.input_frames = motion.shape[0]
        if self.input_frames < 2:
            raise ValueError("Motion must contain at least 2 frames.")

        self._smooth_motion_input()
        self.duration = (self.input_frames - 1) * self.input_dt
        print(f"Motion loaded ({self.motion_file}), duration: {self.duration} sec, frames: {self.input_frames}")

    def _read_soma_csv(self) -> np.ndarray:
        with open(self.motion_file, newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames is None:
                raise ValueError(f"Input CSV has no header: {self.motion_file}")

            missing = [name for name in SOMA_REQUIRED_COLUMNS if name not in reader.fieldnames]
            if missing:
                raise ValueError(f"Soma CSV is missing required columns: {missing}")

            rows = list(reader)

        if self.frame_range is not None:
            start, end = self.frame_range
            if start < 1 or end < start:
                raise ValueError(f"Invalid --frame_range {self.frame_range}; expected 1-based START END.")
            rows = rows[start - 1 : end]

        if not rows:
            raise ValueError(f"No motion rows selected from {self.motion_file}")

        columns = [*SOMA_ROOT_POSITION_COLUMNS, *SOMA_ROOT_EULER_COLUMNS, *SOMA_JOINT_COLUMNS]
        data = np.empty((len(rows), len(columns)), dtype=np.float32)
        for row_idx, row in enumerate(rows):
            for col_idx, column in enumerate(columns):
                data[row_idx, col_idx] = float(row[column])

        return data

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
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0], self.motion_base_rots_input[index_1], blend
        )
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
    return f"{output_stem}.csv"


def run_simulator(sim: SimulationContext, scene: InteractiveScene):
    motion = MotionLoader(
        motion_file=args_cli.input_file,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=args_cli.frame_range,
        position_scale=args_cli.position_scale,
        angle_unit=args_cli.angle_unit,
        root_height_offset=args_cli.root_height_offset,
        smooth_window=args_cli.smooth_window,
        smooth_passes=args_cli.smooth_passes,
        smooth_fields=tuple(args_cli.smooth_fields),
    )

    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(K1_REV1_MOTION_CSV_JOINT_NAMES, preserve_order=True)[0]
    if len(robot_joint_indexes) != len(K1_REV1_MOTION_CSV_JOINT_NAMES):
        raise RuntimeError("Could not resolve all K1 Rev.1 23-DoF joints in the robot asset.")

    motion_base_rot_xyzw = motion.motion_base_rots[:, [1, 2, 3, 0]]
    motion_csv = torch.cat((motion.motion_base_poss, motion_base_rot_xyzw, motion.motion_dof_poss), dim=1)
    output_csv_path = _get_output_csv_path(args_cli.output_name)
    np.savetxt(output_csv_path, motion_csv.cpu().numpy(), delimiter=",", fmt="%.9f")
    print("[INFO]: Mimic-style motion csv file saved to", output_csv_path)
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
