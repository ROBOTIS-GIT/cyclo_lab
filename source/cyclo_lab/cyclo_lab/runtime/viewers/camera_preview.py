"""Non-blocking RGB camera preview backed by POSIX shared memory."""

from __future__ import annotations

import atexit
import json
import os
import struct
import subprocess
import threading
from multiprocessing import shared_memory
from pathlib import Path
from queue import Empty, Full, Queue

import numpy as np
import torch

_FRAME_OFFSET = 64
_SEQUENCE = struct.Struct("<Q")
_STOP_SEQUENCE = (1 << 64) - 1
_STOP_FRAME = object()


class SharedMemoryCameraPreview:
    """Send the latest RGB frame to an isolated lightweight viewer process."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        window_title: str = "SG2 Overhead Camera",
        window_size: int = 720,
        window_position: tuple[int, int] | None = None,
        panel_labels: tuple[tuple[str, int, int], ...] = (),
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self._sequence = 0
        self._closed = False
        self._shared_memory = shared_memory.SharedMemory(
            create=True,
            size=_FRAME_OFFSET + self.width * self.height * 3,
        )
        self._frame = np.ndarray(
            (self.height, self.width, 3),
            dtype=np.uint8,
            buffer=self._shared_memory.buf,
            offset=_FRAME_OFFSET,
        )
        self._frame.fill(0)
        _SEQUENCE.pack_into(self._shared_memory.buf, 0, self._sequence)

        viewer_python = os.environ.get("CYCLO_CAMERA_VIEWER_PYTHON", "/opt/cyclo-camera-viewer/bin/python")
        if not Path(viewer_python).is_file():
            self._cleanup_shared_memory()
            raise RuntimeError(
                f"Camera viewer Python not found: {viewer_python}. Rebuild the cyclo-lab image "
                "or set CYCLO_CAMERA_VIEWER_PYTHON."
            )

        viewer_script = Path(__file__).with_name("camera_preview_process.py")
        viewer_env = os.environ.copy()
        for variable in ("QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH", "PYTHONPATH"):
            viewer_env.pop(variable, None)
        viewer_env.setdefault("QT_X11_NO_MITSHM", "1")
        self._process = subprocess.Popen(
            [
                viewer_python,
                str(viewer_script),
                "--shared-memory",
                self._shared_memory.name,
                "--width",
                str(self.width),
                "--height",
                str(self.height),
                "--window-size",
                str(int(window_size)),
                "--title",
                window_title,
                "--parent-pid",
                str(os.getpid()),
                "--panel-labels",
                json.dumps(panel_labels),
                *(
                    ["--window-x", str(window_position[0]), "--window-y", str(window_position[1])]
                    if window_position is not None
                    else []
                ),
            ],
            env=viewer_env,
            start_new_session=True,
        )
        atexit.register(self.close)

    @property
    def is_open(self) -> bool:
        return not self._closed and self._process.poll() is None

    def update(self, image) -> None:
        """Publish one RGB frame, replacing any frame the viewer has not consumed."""
        if not self.is_open:
            return
        if hasattr(image, "detach"):
            image = image.detach()
            if getattr(image, "ndim", 0) == 4:
                image = image[0]
            image = image.to(device="cpu", copy=True).contiguous().numpy()
        else:
            image = np.asarray(image)

        if image.ndim != 3 or image.shape[0:2] != (self.height, self.width):
            raise ValueError(f"Preview frame must have shape ({self.height}, {self.width}, C), got {image.shape}.")
        if image.shape[2] < 3:
            raise ValueError(f"Preview frame must have at least 3 channels, got {image.shape[2]}.")
        image = image[..., :3]
        if image.dtype != np.uint8:
            if image.size and float(image.max()) <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)

        _SEQUENCE.pack_into(self._shared_memory.buf, 0, self._sequence + 1)
        np.copyto(self._frame, image)
        self._sequence += 2
        _SEQUENCE.pack_into(self._shared_memory.buf, 0, self._sequence)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process.poll() is None:
            _SEQUENCE.pack_into(self._shared_memory.buf, 0, _STOP_SEQUENCE)
            try:
                self._process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    self._process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=1.0)
        self._cleanup_shared_memory()
        try:
            atexit.unregister(self.close)
        except Exception:
            pass

    def _cleanup_shared_memory(self) -> None:
        try:
            self._shared_memory.close()
        except Exception:
            pass
        try:
            self._shared_memory.unlink()
        except FileNotFoundError:
            pass


def _camera_rgb(env, camera_name: str) -> torch.Tensor:
    image = env.scene.sensors[camera_name].data.output["rgb"]
    if image.ndim == 4:
        image = image[0]
    return image[..., :3]


class CameraDashboard:
    """Compose native-resolution Isaac Lab camera tensors into one preview."""

    def __init__(
        self,
        env,
        *,
        rows: tuple[tuple[tuple[str, str], ...], ...],
        panel_rotations: dict[str, int] | None = None,
        gap: int = 8,
        window_title: str = "Camera Dashboard",
        window_size: int = 1800,
        window_position: tuple[int, int] | None = None,
    ) -> None:
        if not rows or any(not row for row in rows):
            raise ValueError("Camera dashboard rows must not be empty.")
        self._env = env
        self._panel_rotations = {
            camera_name: int(quarter_turns) % 4 for camera_name, quarter_turns in (panel_rotations or {}).items()
        }
        panel_sizes = {}
        first_image = None
        for row in rows:
            for camera_name, _ in row:
                image = _camera_rgb(env, camera_name)
                if first_image is None:
                    first_image = image
                elif image.device != first_image.device or image.dtype != first_image.dtype:
                    raise ValueError("Dashboard cameras must use the same device and pixel dtype.")
                image_height, image_width = int(image.shape[0]), int(image.shape[1])
                if self._panel_rotations.get(camera_name, 0) % 2:
                    image_height, image_width = image_width, image_height
                panel_sizes[camera_name] = (image_width, image_height)

        row_widths = [sum(panel_sizes[camera_name][0] for camera_name, _ in row) + gap * (len(row) - 1) for row in rows]
        row_heights = [max(panel_sizes[camera_name][1] for camera_name, _ in row) for row in rows]
        self.width = max(row_widths)
        self.height = sum(row_heights) + gap * (len(rows) - 1)
        background = 18 if not first_image.dtype.is_floating_point else 18.0 / 255.0
        self._frame = torch.full(
            (self.height, self.width, 3),
            background,
            dtype=first_image.dtype,
            device=first_image.device,
        )

        self._layout = {}
        labels = []
        row_y = 0
        for row, row_width, row_height in zip(rows, row_widths, row_heights):
            panel_x = (self.width - row_width) // 2
            for camera_name, label in row:
                panel_width, panel_height = panel_sizes[camera_name]
                panel_y = row_y + (row_height - panel_height) // 2
                self._layout[camera_name] = (panel_x, panel_y, panel_width, panel_height)
                labels.append((label, panel_x, panel_y))
                panel_x += panel_width + gap
            row_y += row_height + gap

        self._preview = SharedMemoryCameraPreview(
            width=self.width,
            height=self.height,
            window_title=window_title,
            window_size=min(self.width, int(window_size)),
            window_position=window_position,
            panel_labels=tuple(labels),
        )
        self._frame_queue: Queue = Queue(maxsize=1)
        self._buffer_pool: Queue = Queue(maxsize=2)
        for _ in range(2):
            self._buffer_pool.put(
                torch.empty_like(
                    self._frame,
                    device="cpu",
                    pin_memory=self._frame.is_cuda,
                )
            )
        self._closed = False
        self._warned_queue_full = False
        self._worker = threading.Thread(
            target=self._preview_loop,
            name="camera-dashboard-preview",
            daemon=True,
        )
        self._worker.start()

    def update(self) -> None:
        if self._closed or not self._preview.is_open:
            return
        if self._frame_queue.full():
            self._warn_saturated()
            return
        try:
            frame_cpu = self._buffer_pool.get_nowait()
        except Empty:
            self._warn_saturated()
            return

        try:
            for camera_name, (panel_x, panel_y, panel_width, panel_height) in self._layout.items():
                image = _camera_rgb(self._env, camera_name)
                quarter_turns = self._panel_rotations.get(camera_name, 0)
                if quarter_turns:
                    image = torch.rot90(image, k=quarter_turns, dims=(0, 1))
                if image.shape[:2] != (panel_height, panel_width):
                    raise ValueError(
                        f"Camera {camera_name} changed size from {panel_width}x{panel_height} "
                        f"to {image.shape[1]}x{image.shape[0]}."
                    )
                self._frame[
                    panel_y : panel_y + panel_height,
                    panel_x : panel_x + panel_width,
                ].copy_(image)

            frame_cpu.copy_(self._frame, non_blocking=self._frame.is_cuda)
            if self._frame.is_cuda:
                ready_event = torch.cuda.Event()
                ready_event.record(torch.cuda.current_stream(self._frame.device))
            else:
                ready_event = None
            self._frame_queue.put_nowait((frame_cpu, ready_event))
            frame_cpu = None
        except Full:
            self._warn_saturated()
        finally:
            if frame_cpu is not None:
                self._buffer_pool.put_nowait(frame_cpu)

    def _preview_loop(self) -> None:
        while True:
            try:
                item = self._frame_queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                if item is _STOP_FRAME:
                    return
                frame, ready_event = item
                if ready_event is not None:
                    ready_event.synchronize()
                self._preview.update(frame.numpy())
            finally:
                if item is not _STOP_FRAME:
                    self._buffer_pool.put(item[0])
                self._frame_queue.task_done()

    def _warn_saturated(self) -> None:
        if self._warned_queue_full:
            return
        self._warned_queue_full = True
        print("[Camera Preview] dashboard is saturated; dropping the newest frame.")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._frame_queue.join()
        try:
            self._frame_queue.put_nowait(_STOP_FRAME)
        except Full:
            self._frame_queue.put(_STOP_FRAME)
        self._worker.join(timeout=2.0)
        if self._worker.is_alive():
            print("[Camera Preview] dashboard worker did not stop within 2 seconds.")
        self._preview.close()
