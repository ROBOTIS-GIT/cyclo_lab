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

"""Kinematics-only replay for normalized K1 Rev.1 motion CSV files."""

import argparse
import os
import sys
import time
import traceback

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


parser = argparse.ArgumentParser(description="Kinematics-only replay for normalized K1 Rev.1 motion CSV files.")
parser.add_argument("--input_file", "-f", type=str, required=True, help="Normalized motion CSV file.")
parser.add_argument("--fps", type=float, default=50.0, help="Replay FPS.")
parser.add_argument(
    "--root-quat-order",
    type=str,
    default="xyzw",
    choices=("xyzw", "wxyz"),
    help="Order of the root quaternion columns in the CSV.",
)
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help="1-based inclusive frame range. If omitted, all frames are used.",
)
parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
parser.add_argument("--loop", action="store_true", help="Loop the motion until the app is closed.")
parser.add_argument("--max-frames", type=int, help="Maximum rendered frames. Useful for headless smoke tests.")
parser.add_argument("--no-camera-follow", action="store_true", help="Keep the default camera instead of following root.")
parser.add_argument(
    "--camera-offset",
    nargs=3,
    type=float,
    default=(2.0, 2.0, 0.6),
    metavar=("X", "Y", "Z"),
    help="Camera offset from the root when camera follow is enabled.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from cyclo_lab.assets.robots.K1_rev1 import K1_REV1_INERTIA_TUNED_CFG


@configclass
class ReplaySceneCfg(InteractiveSceneCfg):
    """Minimal scene for direct kinematic replay."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = K1_REV1_INERTIA_TUNED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _load_csv(path: str, root_quat_order: str, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if args_cli.frame_range is None:
        motion_np = np.loadtxt(path, delimiter=",")
    else:
        start, end = args_cli.frame_range
        if start < 1 or end < start:
            raise ValueError(f"Invalid --frame_range {args_cli.frame_range}; expected 1-based START END.")
        motion_np = np.loadtxt(path, delimiter=",", skiprows=start - 1, max_rows=end - start + 1)

    if motion_np.ndim == 1:
        motion_np = motion_np[None, :]
    if motion_np.shape[1] != 30:
        raise ValueError(f"Expected 30 CSV columns: root xyz + root quat + 23 joints, got {motion_np.shape[1]}.")

    motion = torch.as_tensor(motion_np, dtype=torch.float32, device=device)
    root_pos = motion[:, :3]
    root_quat = motion[:, 3:7]
    if root_quat_order == "xyzw":
        root_quat = root_quat[:, [3, 0, 1, 2]]
    root_quat_norm = torch.sqrt(torch.clamp(torch.sum(root_quat * root_quat, dim=1, keepdim=True), min=1.0e-8))
    root_quat = root_quat / root_quat_norm
    joint_pos = motion[:, 7:]
    return root_pos, root_quat, joint_pos


def _sleep_for_realtime(start_time: float, dt: float):
    if args_cli.speed <= 0.0:
        return
    elapsed = time.perf_counter() - start_time
    delay = dt / args_cli.speed - elapsed
    if delay > 0.0:
        time.sleep(delay)


def run_replay(sim: SimulationContext, scene: InteractiveScene):
    root_pos, root_quat, motion_joint_pos = _load_csv(args_cli.input_file, args_cli.root_quat_order, sim.device)
    dt = 1.0 / args_cli.fps
    robot = scene["robot"]
    joint_ids = robot.find_joints(K1_REV1_MOTION_CSV_JOINT_NAMES, preserve_order=True)[0]
    if len(joint_ids) != len(K1_REV1_MOTION_CSV_JOINT_NAMES):
        raise RuntimeError("Could not resolve all K1 Rev.1 CSV joints in the robot asset.")

    print(f"[INFO] Replaying CSV: {args_cli.input_file}", flush=True)
    print(f"[INFO] Frames: {root_pos.shape[0]}, fps: {args_cli.fps}, loop: {args_cli.loop}", flush=True)

    frame_id = 0
    rendered_frames = 0
    camera_offset = np.asarray(args_cli.camera_offset)
    while simulation_app.is_running():
        step_start = time.perf_counter()

        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] = root_pos[frame_id]
        root_state[:, :2] += scene.env_origins[:, :2]
        root_state[:, 3:7] = root_quat[frame_id]
        root_state[:, 7:] = 0.0
        robot.write_root_state_to_sim(root_state)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, joint_ids] = motion_joint_pos[frame_id]
        joint_vel[:, :] = 0.0
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        sim.render()
        scene.update(dt)

        if not args_cli.no_camera_follow:
            lookat = root_state[0, :3].cpu().numpy()
            sim.set_camera_view(lookat + camera_offset, lookat)

        rendered_frames += 1
        if args_cli.max_frames is not None and rendered_frames >= args_cli.max_frames:
            break

        frame_id += 1
        if frame_id >= root_pos.shape[0]:
            if not args_cli.loop:
                break
            frame_id = 0

        _sleep_for_realtime(step_start, dt)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.fps
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(ReplaySceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    print(
        "[INFO] Setup complete. This is kinematics-only replay; no policy, actuator control, or physics stepping.",
        flush=True,
    )
    run_replay(sim, scene)
    print("[INFO] Replay complete.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] Replay failed: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        os._exit(1)
    os._exit(0)
