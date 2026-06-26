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

"""Kinematics-only replay for K1 Rev.1 motion NPZ files."""

import argparse
import os
import sys
import time
import traceback

import numpy as np

from isaaclab.app import AppLauncher


REQUIRED_NPZ_KEYS = (
    "fps",
    "joint_pos",
    "body_pos_w",
    "body_quat_w",
)


parser = argparse.ArgumentParser(description="Kinematics-only replay for K1 Rev.1 motion NPZ files.")
parser.add_argument("--input_file", "-f", type=str, required=True, help="Motion NPZ file.")
parser.add_argument("--fps", type=float, help="Replay FPS. Defaults to the NPZ fps key.")
parser.add_argument(
    "--root-body-index",
    type=int,
    default=0,
    help="Body index used as the robot root pose in body_pos_w/body_quat_w.",
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


def _load_npz(path: str, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    data = np.load(path)
    missing = [key for key in REQUIRED_NPZ_KEYS if key not in data]
    if missing:
        raise KeyError(f"NPZ file is missing required keys: {missing}")

    joint_pos_np = data["joint_pos"]
    body_pos_np = data["body_pos_w"]
    body_quat_np = data["body_quat_w"]
    if joint_pos_np.ndim != 2:
        raise ValueError(f"joint_pos must be rank 2, got shape {joint_pos_np.shape}.")
    if body_pos_np.ndim != 3 or body_pos_np.shape[-1] != 3:
        raise ValueError(f"body_pos_w must have shape (frames, bodies, 3), got {body_pos_np.shape}.")
    if body_quat_np.ndim != 3 or body_quat_np.shape[-1] != 4:
        raise ValueError(f"body_quat_w must have shape (frames, bodies, 4), got {body_quat_np.shape}.")
    if not (0 <= args_cli.root_body_index < body_pos_np.shape[1]):
        raise ValueError(f"--root-body-index is out of range for {body_pos_np.shape[1]} bodies.")

    if args_cli.frame_range is not None:
        start, end = args_cli.frame_range
        if start < 1 or end < start:
            raise ValueError(f"Invalid --frame_range {args_cli.frame_range}; expected 1-based START END.")
        frame_slice = slice(start - 1, end)
        joint_pos_np = joint_pos_np[frame_slice]
        body_pos_np = body_pos_np[frame_slice]
        body_quat_np = body_quat_np[frame_slice]

    fps = float(args_cli.fps if args_cli.fps is not None else np.asarray(data["fps"]).reshape(-1)[0])
    root_pos = torch.as_tensor(body_pos_np[:, args_cli.root_body_index], dtype=torch.float32, device=device)
    root_quat = torch.as_tensor(body_quat_np[:, args_cli.root_body_index], dtype=torch.float32, device=device)
    root_quat_norm = torch.sqrt(torch.clamp(torch.sum(root_quat * root_quat, dim=1, keepdim=True), min=1.0e-8))
    root_quat = root_quat / root_quat_norm
    joint_pos = torch.as_tensor(joint_pos_np, dtype=torch.float32, device=device)
    return root_pos, root_quat, joint_pos, fps


def _sleep_for_realtime(start_time: float, dt: float):
    if args_cli.speed <= 0.0:
        return
    elapsed = time.perf_counter() - start_time
    delay = dt / args_cli.speed - elapsed
    if delay > 0.0:
        time.sleep(delay)


def run_replay(sim: SimulationContext, scene: InteractiveScene):
    root_pos, root_quat, motion_joint_pos, fps = _load_npz(args_cli.input_file, sim.device)
    dt = 1.0 / fps
    robot = scene["robot"]
    if motion_joint_pos.shape[1] != robot.data.default_joint_pos.shape[1]:
        raise ValueError(
            f"NPZ joint_pos has {motion_joint_pos.shape[1]} joints, "
            f"but robot has {robot.data.default_joint_pos.shape[1]} joints."
        )

    print(f"[INFO] Replaying NPZ: {args_cli.input_file}", flush=True)
    print(f"[INFO] Frames: {root_pos.shape[0]}, fps: {fps}, loop: {args_cli.loop}", flush=True)

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
        joint_pos[:, :] = motion_joint_pos[frame_id]
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
    root_pos, _, _, fps = _load_npz(args_cli.input_file, "cpu")
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / fps
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(ReplaySceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    print(f"[INFO] Setup complete for {root_pos.shape[0]} frames.", flush=True)
    print("[INFO] This is kinematics-only replay; no policy, actuator control, or physics stepping.", flush=True)
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
