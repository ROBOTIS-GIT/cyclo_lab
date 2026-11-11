"""OpenCV process for displaying a shared-memory RGB camera frame."""

from __future__ import annotations

import argparse
import json
import os
import struct
import time
from multiprocessing import resource_tracker, shared_memory

import cv2
import numpy as np

_FRAME_OFFSET = 64
_SEQUENCE = struct.Struct("<Q")
_STOP_SEQUENCE = (1 << 64) - 1


def _parent_is_alive(parent_pid: int) -> bool:
    try:
        os.kill(parent_pid, 0)
        return True
    except ProcessLookupError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Display the latest RGB frame from shared memory.")
    parser.add_argument("--shared-memory", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--window-size", type=int, default=720)
    parser.add_argument("--title", default="Camera Preview")
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--window-x", type=int)
    parser.add_argument("--window-y", type=int)
    parser.add_argument("--panel-labels", default="[]")
    args = parser.parse_args()
    panel_labels = json.loads(args.panel_labels)

    memory = shared_memory.SharedMemory(name=args.shared_memory)
    resource_tracker.unregister(memory._name, "shared_memory")
    frame_view = np.ndarray(
        (args.height, args.width, 3),
        dtype=np.uint8,
        buffer=memory.buf,
        offset=_FRAME_OFFSET,
    )

    cv2.namedWindow(args.title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    window_scale = args.window_size / max(args.width, args.height)
    cv2.resizeWindow(
        args.title,
        max(1, round(args.width * window_scale)),
        max(1, round(args.height * window_scale)),
    )
    if args.window_x is not None and args.window_y is not None:
        cv2.moveWindow(args.title, args.window_x, args.window_y)
    last_sequence = 0
    try:
        while _parent_is_alive(args.parent_pid):
            sequence_before = _SEQUENCE.unpack_from(memory.buf, 0)[0]
            if sequence_before == _STOP_SEQUENCE:
                break
            if sequence_before != last_sequence and sequence_before % 2 == 0:
                frame = frame_view.copy()
                sequence_after = _SEQUENCE.unpack_from(memory.buf, 0)[0]
                if sequence_before == sequence_after:
                    display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    for label, panel_x, panel_y in panel_labels:
                        text_origin = (int(panel_x) + 12, int(panel_y) + 28)
                        (text_width, text_height), baseline = cv2.getTextSize(
                            label,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            2,
                        )
                        cv2.rectangle(
                            display_frame,
                            (text_origin[0] - 7, text_origin[1] - text_height - 7),
                            (text_origin[0] + text_width + 7, text_origin[1] + baseline + 7),
                            (24, 24, 24),
                            thickness=-1,
                        )
                        cv2.putText(
                            display_frame,
                            label,
                            text_origin,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (245, 245, 245),
                            2,
                            cv2.LINE_AA,
                        )
                    cv2.imshow(args.title, display_frame)
                    last_sequence = sequence_after

            key = cv2.waitKey(10) & 0xFF
            if key in (27, ord("q")):
                break
            if sequence_before == last_sequence:
                time.sleep(0.005)
    finally:
        cv2.destroyAllWindows()
        memory.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
