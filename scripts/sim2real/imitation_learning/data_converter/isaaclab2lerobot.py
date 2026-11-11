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

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CYCLO_LAB_SOURCE = _REPO_ROOT / "source" / "cyclo_lab"
if _CYCLO_LAB_SOURCE.is_dir():
    sys.path.insert(0, str(_CYCLO_LAB_SOURCE))

from cyclo_lab.robot_specs.ffw.sg2 import FFW_SG2_PUBLISHED_JOINT_NAMES
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROBOT_CONFIGS = {
    "OMY": {
        "expected_dim": 7,
        "joint_names": [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "rh_r1_joint",
        ],
        "cameras": {
            "cam_wrist": {"height": 480, "width": 848},
            "cam_top": {"height": 480, "width": 848},
        },
    },
    "FFW_SG2": {
        "expected_dim": 19,
        "joint_names": list(FFW_SG2_PUBLISHED_JOINT_NAMES),
        "cameras": {
            "cam_head": {"height": 376, "width": 672},
        },
    },
}


def _ffw_sg2_action_to_lerobot(actions: np.ndarray) -> np.ndarray:
    """Convert Isaac Lab SG2 actions into the published/LeRobot joint order."""
    if actions.ndim != 2 or actions.shape[1] != 19:
        raise ValueError(f"FFW_SG2 actions must have shape [N, 19], got {tuple(actions.shape)}.")

    # HDF5 actions follow Isaac Lab action-term order:
    # [arm_l, gripper_l, arm_r, gripper_r, lift, head(2)].
    # WoodBlock LeRobot datasets use:
    # [arm_l, gripper_l, arm_r, gripper_r, head(2), lift].
    return np.concatenate(
        [
            actions[:, :16],
            actions[:, 17:19],
            actions[:, 16:17],
        ],
        axis=-1,
    )


def get_env_features(fps: int, robot_type: str, camera_shapes: dict[str, dict[str, int]] | None = None):
    if robot_type not in ROBOT_CONFIGS:
        raise ValueError(f"Unsupported robot type: {robot_type}")

    config = ROBOT_CONFIGS[robot_type]
    camera_shapes = camera_shapes or {}

    # Build action and observation.state features
    features = {
        "action": {
            "dtype": "float32",
            "shape": (config["expected_dim"],),
            "names": config["joint_names"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (config["expected_dim"],),
            "names": config["joint_names"],
        },
    }

    # Add camera features
    for cam_name, cam_cfg in config["cameras"].items():
        cam_shape = camera_shapes.get(cam_name, {})
        height = int(cam_shape.get("height", cam_cfg["height"]))
        width = int(cam_shape.get("width", cam_cfg["width"]))
        features[f"observation.images.{cam_name}"] = {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channels"],
            "video_info": {
                "video.height": height,
                "video.width": width,
                "video.codec": "libx264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": 3,
                "has_audio": False,
            },
        }

    return features


def _infer_camera_shapes_from_hdf5(dataset_file: str, robot_type: str) -> dict[str, dict[str, int]]:
    """Infer recorded camera image sizes from the first valid HDF5 demo."""
    if robot_type not in ROBOT_CONFIGS:
        raise ValueError(f"Unsupported robot type: {robot_type}")

    camera_shapes: dict[str, dict[str, int]] = {}
    camera_keys = list(ROBOT_CONFIGS[robot_type]["cameras"].keys())
    with h5py.File(dataset_file, "r") as f:
        if "data" not in f:
            return camera_shapes
        for demo_name in f["data"].keys():
            demo_group = f["data"][demo_name]
            for cam_key in camera_keys:
                dataset_key = f"obs/{cam_key}"
                if dataset_key not in demo_group:
                    continue
                shape = demo_group[dataset_key].shape
                if len(shape) < 4:
                    continue
                if shape[-1] not in (1, 3, 4):
                    continue
                camera_shapes[cam_key] = {"height": int(shape[-3]), "width": int(shape[-2])}
            if camera_shapes:
                break

    return camera_shapes


def _read_timestamps(demo_group: h5py.Group) -> np.ndarray | None:
    """Read per-frame wall-clock timestamps from a recorded HDF5 demo."""
    for key in ("obs/timestamp", "obs/wall_time", "timestamp"):
        if key in demo_group:
            timestamps = np.asarray(demo_group[key], dtype=np.float64).reshape(-1)
            if timestamps.size > 0:
                return timestamps
    return None


def _select_frame_indices(
    total_frames: int,
    frame_skip: int,
    frame_stride: int,
    fps: int,
    timestamps: np.ndarray | None,
    resample_by_time: bool,
    demo_name: str,
) -> list[int]:
    """Select source frame indices using either fixed stride or timestamp-based resampling."""
    if frame_skip < 0:
        raise ValueError(f"frame_skip must be >= 0, got {frame_skip}")
    if frame_stride < 1:
        raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    source_indices = np.arange(frame_skip, total_frames, frame_stride, dtype=np.int64)
    if source_indices.size == 0:
        return []

    if not resample_by_time:
        return source_indices.tolist()

    if timestamps is None:
        print(f"[WARN] Demo {demo_name} has no timestamps; falling back to frame_stride={frame_stride}.")
        return source_indices.tolist()

    if timestamps.shape[0] < total_frames:
        print(
            f"[WARN] Demo {demo_name} timestamp length ({timestamps.shape[0]}) is shorter than frame count "
            f"({total_frames}); falling back to frame_stride={frame_stride}."
        )
        return source_indices.tolist()

    source_timestamps = timestamps[source_indices]
    finite_mask = np.isfinite(source_timestamps)
    if not finite_mask.all():
        source_indices = source_indices[finite_mask]
        source_timestamps = source_timestamps[finite_mask]
    if source_indices.size == 0:
        return []

    # Normalize per episode, then enforce monotonic order. If timestamps are not
    # monotonic enough to resample, keep the older frame-stride behavior.
    source_timestamps = source_timestamps - source_timestamps[0]
    if source_timestamps.size < 2 or source_timestamps[-1] <= 0.0:
        print(f"[WARN] Demo {demo_name} timestamps have no positive duration; falling back to frame_stride.")
        return source_indices.tolist()
    if np.any(np.diff(source_timestamps) < 0.0):
        print(f"[WARN] Demo {demo_name} timestamps are not monotonic; falling back to frame_stride.")
        return source_indices.tolist()

    target_period = 1.0 / fps
    target_timestamps = np.arange(0.0, source_timestamps[-1] + 0.5 * target_period, target_period)
    right = np.searchsorted(source_timestamps, target_timestamps, side="left")
    right = np.clip(right, 0, source_timestamps.shape[0] - 1)
    left = np.clip(right - 1, 0, source_timestamps.shape[0] - 1)
    choose_left = np.abs(source_timestamps[left] - target_timestamps) <= np.abs(
        source_timestamps[right] - target_timestamps
    )
    selected_offsets = np.where(choose_left, left, right)
    selected_indices = source_indices[selected_offsets]

    measured_hz = (source_timestamps.size - 1) / source_timestamps[-1]
    print(
        f"[INFO] Demo {demo_name}: timestamp resample {source_timestamps.size} frames "
        f"({measured_hz:.2f} Hz measured) -> {selected_indices.size} frames at {fps} Hz."
    )
    return selected_indices.astype(np.int64).tolist()


def process_data(
    dataset: LeRobotDataset,
    task: str,
    demo_group: h5py.Group,
    demo_name: str,
    frame_skip: int,
    frame_stride: int,
    fps: int,
    resample_by_time: bool,
    robot_type: str,
) -> bool:
    """
    Process a single demonstration group from the HDF5 dataset
    and add it into the LeRobot dataset.
    """
    if robot_type not in ROBOT_CONFIGS:
        raise ValueError(f"Unsupported robot type: {robot_type}")

    config = ROBOT_CONFIGS[robot_type]
    camera_items = list(config["cameras"].items())

    try:
        # Load action and state data
        actions = np.array(demo_group["actions"], dtype=np.float32)
        joint_pos = np.array(demo_group["obs/joint_pos"], dtype=np.float32)
        # Keep camera datasets lazy to avoid loading every image into RAM at once.
        camera_data = {}
        for cam_key, _cam_cfg in camera_items:
            camera_data[f"observation.images.{cam_key}"] = demo_group[f"obs/{cam_key}"]

    except KeyError as e:
        print(f"Demo {demo_name} is not valid (missing key: {e}), skipping...")
        return False

    if actions.shape[0] < 10:
        print(f"Demo {demo_name} has insufficient frames ({actions.shape[0]}), skipping...")
        return False

    # Ensure actions and joint positions are 2D arrays
    if actions.ndim == 1:
        actions = actions.reshape(-1, config["expected_dim"])
    if joint_pos.ndim == 1:
        joint_pos = joint_pos.reshape(-1, config["expected_dim"])
    if robot_type == "FFW_SG2":
        actions = _ffw_sg2_action_to_lerobot(actions)

    total_state_frames = actions.shape[0]

    timestamps = _read_timestamps(demo_group)
    frame_indices = _select_frame_indices(
        total_frames=total_state_frames,
        frame_skip=frame_skip,
        frame_stride=frame_stride,
        fps=fps,
        timestamps=timestamps,
        resample_by_time=resample_by_time,
        demo_name=demo_name,
    )
    if not frame_indices:
        print(f"Demo {demo_name} has no selected frames, skipping...")
        return False

    # Process each frame
    for frame_index in tqdm(frame_indices, desc=f"Processing demo {demo_name}"):

        # Build frame dictionary
        frame = {
            "action": actions[frame_index],
            "observation.state": joint_pos[frame_index],
        }

        # Add camera images
        for feature_key, images in camera_data.items():
            frame[feature_key] = images[frame_index]

        dataset.add_frame(frame=frame, task=task)

    return True


def convert_isaaclab_to_lerobot(
    task: str,
    repo_id: str,
    robot_type: str,
    dataset_file: str,
    fps: int,
    push_to_hub: bool = False,
    frame_skip: int = 3,
    frame_stride: int = 1,
    resample_by_time: bool = False,
    root: str = "./datasets/lerobot/sim2real_data",
):
    """
    Convert an IsaacLab HDF5 dataset into LeRobot dataset format.
    """
    hdf5_files = [dataset_file]
    now_episode_index = 0
    camera_shapes = _infer_camera_shapes_from_hdf5(dataset_file, robot_type)
    if camera_shapes:
        shape_text = ", ".join(
            f"{name}={shape['width']}x{shape['height']}" for name, shape in sorted(camera_shapes.items())
        )
        print(f"[INFO] Inferred camera feature shapes from HDF5: {shape_text}")

    # Create a new LeRobot dataset
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=get_env_features(fps, robot_type, camera_shapes=camera_shapes),
        root=root,
    )

    # Process each HDF5 dataset file
    for hdf5_id, hdf5_file in enumerate(hdf5_files):
        print(f"[{hdf5_id + 1}/{len(hdf5_files)}] Processing HDF5 file: {hdf5_file}")
        with h5py.File(hdf5_file, "r") as f:
            demo_names = list(f["data"].keys())
            print(f"Found {len(demo_names)} demos: {demo_names}")

            for demo_name in tqdm(demo_names, desc="Processing each demo"):
                demo_group = f["data"][demo_name]

                # Skip unsuccessful demonstrations
                if "success" in demo_group.attrs and not demo_group.attrs["success"]:
                    print(f"Demo {demo_name} not successful, skipping...")
                    continue

                valid = process_data(
                    dataset,
                    task,
                    demo_group,
                    demo_name,
                    frame_skip,
                    frame_stride,
                    fps,
                    resample_by_time,
                    robot_type,
                )

                if valid:
                    now_episode_index += 1
                    dataset.save_episode()
                    print(f"Saved episode {now_episode_index} successfully")

    # Optionally push to HuggingFace Hub
    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert IsaacLab dataset to LeRobot format")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g., OMY_Pickup)")
    parser.add_argument("--robot_type", type=str, default="OMY", help="Robot type (default: OMY)")
    parser.add_argument("--dataset_file", type=str, default="./datasets/dataset.hdf5", help="Path to dataset HDF5 file")
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help=(
            "Target LeRobot dataset rate in Hz. With --resample_by_time, timestamps are resampled "
            "onto this rate (default: 10)."
        ),
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push dataset to HuggingFace Hub")
    parser.add_argument(
        "--frame_skip",
        type=int,
        default=2,
        help="Number of initial frames to drop from each demo (default: 2)",
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Keep every Nth frame after frame_skip. Use 4 to convert 60Hz recordings to 15Hz (default: 1)",
    )
    parser.add_argument(
        "--resample_by_time",
        action="store_true",
        help="Use HDF5 timestamps to resample onto the requested --fps time grid when timestamps are available.",
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    default_repo_id = f"./datasets/lerobot/{timestamp}"
    parser.add_argument("--repo_id", type=str, default=default_repo_id, help=f"Repo ID (default: {default_repo_id})")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help=(
            "Local LeRobot dataset root. Defaults to repo_id when repo_id is a local path, otherwise a timestamped"
            " path."
        ),
    )

    args = parser.parse_args()
    root = args.root
    if root is None:
        root = args.repo_id if args.repo_id.startswith((".", "/")) else default_repo_id

    convert_isaaclab_to_lerobot(
        task=args.task,
        repo_id=args.repo_id,
        robot_type=args.robot_type,
        dataset_file=args.dataset_file,
        fps=args.fps,
        push_to_hub=args.push_to_hub,
        frame_skip=args.frame_skip,
        frame_stride=args.frame_stride,
        resample_by_time=args.resample_by_time,
        root=root,
    )
