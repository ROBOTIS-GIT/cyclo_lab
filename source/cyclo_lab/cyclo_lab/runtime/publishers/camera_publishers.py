"""Compressed image publishing helpers for IsaacLab camera sensors."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from queue import Empty, Full, Queue

import cv2
import torch
from cyclo_lab.runtime.transport.ros2_zenoh import (
    COMPRESSED_IMAGE,
    create_publisher,
    make_compressed_image_kwargs,
    now_time_msg,
)


def publish_compressed_camera(
    camera_name: str,
    camera,
    writer,
    *,
    frame_id: str | None = None,
    stamp_fn: Callable | None = None,
    image_rotation_quarter_turns: int = 0,
) -> None:
    image, ready_event = _camera_rgb_cpu(camera)
    if ready_event is not None:
        ready_event.synchronize()
    img = _rgb_tensor_to_numpy(image, image_rotation_quarter_turns)
    _publish_compressed_rgb(
        camera_name,
        img,
        writer,
        frame_id=frame_id,
        stamp=stamp_fn() if stamp_fn is not None else now_time_msg(),
    )


def _camera_rgb_cpu(camera, output_buffer=None):
    """Start a non-blocking copy of one rendered camera frame to pinned CPU memory."""
    img = camera.data.output["rgb"][0].detach()
    if img.ndim != 3 or img.shape[-1] not in (3, 4) or img.numel() == 0:
        raise RuntimeError(f"camera RGB tensor is not ready: shape={tuple(img.shape)}")
    if not img.is_cuda:
        return img.contiguous().cpu(), None

    image_cpu = output_buffer
    if image_cpu is None:
        image_cpu = torch.empty_like(img, device="cpu", pin_memory=True)
    elif image_cpu.shape != img.shape or image_cpu.dtype != img.dtype:
        raise ValueError(
            f"camera CPU buffer mismatch: expected {tuple(img.shape)} {img.dtype}, "
            f"got {tuple(image_cpu.shape)} {image_cpu.dtype}"
        )
    image_cpu.copy_(img, non_blocking=True)
    ready_event = torch.cuda.Event()
    ready_event.record(torch.cuda.current_stream(img.device))
    return image_cpu, ready_event


def _rgb_tensor_to_numpy(img, image_rotation_quarter_turns: int = 0):
    quarter_turns = int(image_rotation_quarter_turns) % 4
    if quarter_turns:
        img = torch.rot90(img, k=quarter_turns, dims=(0, 1))
    img = img.contiguous().numpy()
    if img.dtype != "uint8":
        max_value = float(img.max()) if img.size else 0.0
        if max_value <= 1.0:
            img = img * 255.0
        img = img.clip(0, 255).astype("uint8")
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    return img


def _publish_compressed_rgb(
    camera_name: str,
    img,
    writer,
    *,
    frame_id: str | None,
    stamp,
) -> None:
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(".jpg", img_bgr)
    if not success:
        raise RuntimeError("cv2.imencode('.jpg', image) failed")

    writer.publish(
        **make_compressed_image_kwargs(
            data=buffer.tobytes(),
            frame_id=frame_id or camera_name,
            fmt="jpeg",
            stamp=stamp,
        )
    )


@dataclass(frozen=True)
class _CameraFrame:
    camera_name: str
    image: object
    ready_event: object | None
    rotation_quarter_turns: int
    writer: object
    stamp: object
    buffer_pool: object | None = None


_STOP = object()


class CompressedCameraPublishers:
    """Publish available IsaacLab RGB sensors to ROS2-compatible image topics."""

    def __init__(
        self,
        scene,
        camera_topics: dict[str, str],
        publish_hz: float | None,
        *,
        image_rotations: dict[str, int] | None = None,
    ) -> None:
        self._scene = scene
        self._publish_hz = publish_hz
        self._image_rotations = {
            camera_name: int(quarter_turns) % 4 for camera_name, quarter_turns in (image_rotations or {}).items()
        }
        self._last_publish_time = 0.0
        self._warned_camera_publish_errors: set[str] = set()
        self._warned_queue_full = False
        self._closed = False
        available_cameras = set(scene.sensors)
        if publish_hz == 0.0:
            self.writers = {}
        else:
            self.writers = {
                camera_name: create_publisher(topic, COMPRESSED_IMAGE)
                for camera_name, topic in camera_topics.items()
                if camera_name in available_cameras
            }
        self._queue: Queue = Queue(maxsize=1)
        self._buffer_pools: dict[str, Queue] = {}
        self._worker = None
        if self.writers:
            self._worker = threading.Thread(
                target=self._publish_loop,
                name="compressed-camera-publisher",
                daemon=True,
            )
            self._worker.start()

    @property
    def endpoints(self) -> tuple:
        return tuple(self.writers.values())

    def publish(self) -> None:
        if self._closed or not self.writers:
            return
        if self._publish_hz is not None and self._publish_hz > 0.0:
            now = time.monotonic()
            publish_period = 1.0 / self._publish_hz
            if now - self._last_publish_time < publish_period * 0.95:
                return
            self._last_publish_time = now
        if self._queue.full():
            self._warn_saturated()
            return

        stamp = now_time_msg()
        frames = []
        for camera_name, writer in self.writers.items():
            buffer_pool = self._buffer_pools.setdefault(camera_name, Queue(maxsize=2))
            try:
                output_buffer = buffer_pool.get_nowait()
            except Empty:
                output_buffer = None
            try:
                image, ready_event = _camera_rgb_cpu(
                    self._scene[camera_name],
                    output_buffer=output_buffer,
                )
                frames.append(
                    _CameraFrame(
                        camera_name=camera_name,
                        image=image,
                        ready_event=ready_event,
                        rotation_quarter_turns=self._image_rotations.get(camera_name, 0),
                        writer=writer,
                        stamp=stamp,
                        buffer_pool=buffer_pool if image.is_pinned() else None,
                    )
                )
            except Exception as exc:
                if output_buffer is not None:
                    buffer_pool.put_nowait(output_buffer)
                self._warn_camera_error(camera_name, exc)

        if not frames:
            return
        try:
            self._queue.put_nowait(tuple(frames))
        except Full:
            for frame in frames:
                if frame.buffer_pool is not None:
                    frame.buffer_pool.put_nowait(frame.image)
            self._warn_saturated()

    def _publish_loop(self) -> None:
        while True:
            try:
                batch = self._queue.get(timeout=0.1)
            except Empty:
                continue

            try:
                if batch is _STOP:
                    return
                for frame in batch:
                    try:
                        if frame.ready_event is not None:
                            frame.ready_event.synchronize()
                        _publish_compressed_rgb(
                            frame.camera_name,
                            _rgb_tensor_to_numpy(
                                frame.image,
                                frame.rotation_quarter_turns,
                            ),
                            frame.writer,
                            frame_id=None,
                            stamp=frame.stamp,
                        )
                    except Exception as exc:
                        self._warn_camera_error(frame.camera_name, exc)
                    finally:
                        if frame.buffer_pool is not None:
                            frame.buffer_pool.put(frame.image)
            finally:
                self._queue.task_done()

    def _warn_camera_error(self, camera_name: str, exc: Exception) -> None:
        if camera_name in self._warned_camera_publish_errors:
            return
        self._warned_camera_publish_errors.add(camera_name)
        print(f"[Zenoh ROS2] camera publish error for {camera_name}: {exc}")

    def _warn_saturated(self) -> None:
        if self._warned_queue_full:
            return
        self._warned_queue_full = True
        print("[Zenoh ROS2] camera publisher is saturated; dropping the newest frame batch.")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._worker is None:
            return

        self._queue.join()
        try:
            self._queue.put_nowait(_STOP)
        except Full:
            self._queue.put(_STOP)
        self._worker.join(timeout=2.0)
        if self._worker.is_alive():
            print("[Zenoh ROS2] camera publisher worker did not stop within 2 seconds.")
