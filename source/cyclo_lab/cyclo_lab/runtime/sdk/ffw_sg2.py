"""Legacy SG2 teleoperation SDK built on the shared Zenoh topic bridge."""

from __future__ import annotations

from collections.abc import Callable

from cyclo_lab.robot_specs.ffw.sg2 import FFW_SG2_JOYSTICK_TRIGGER_TOPIC
from cyclo_lab.runtime.bridges.sg2 import FFWSG2TopicBridge
from cyclo_lab.runtime.transport.ros2_zenoh import STRING, create_subscriber


class FFWSG2Sdk(FFWSG2TopicBridge):
    """Preserve B/N/R episode controls for existing SG2 IL scripts."""

    def __init__(
        self,
        env,
        mode: str,
        camera_publish_hz: float | None = None,
        publish_odometry_tf: bool = True,
        enable_joystick_trigger: bool = True,
        enable_keyboard_listener: bool = True,
    ) -> None:
        if mode not in ("record", "inference"):
            raise ValueError(f"Unsupported FFW-SG2 SDK mode: {mode}")
        self.mode = mode
        self._started = False
        self._reset_state = False
        self._additional_callbacks: dict[str, Callable] = {}
        self._episode_phase = "idle"
        self.listener = None

        super().__init__(
            env,
            camera_publish_hz=camera_publish_hz,
            publish_odometry_tf=publish_odometry_tf,
            subscribe_reset=False,
        )

        if enable_joystick_trigger:
            self.subscribers.append(
                create_subscriber(
                    topic=FFW_SG2_JOYSTICK_TRIGGER_TOPIC,
                    msg_type=STRING,
                    callback=self._on_joystick_trigger,
                )
            )
        if enable_keyboard_listener:
            try:
                from pynput.keyboard import Listener

                self.listener = Listener(on_press=self._on_press)
                self.listener.start()
            except ImportError as exc:
                print(f"[Control] Keyboard listener unavailable: {exc}")

        self._print_keyboard_controls()

    def _print_keyboard_controls(self) -> None:
        print("\n[Control] Press keys to control the FFW_SG2 robot:")
        if self.mode == "record":
            print("[B / Right Joystick Button] Start recording the current episode")
            print("[N / Right Joystick Button] Save the current episode")
            print("[R / Left Joystick Button] Skip and reset the current episode")
            print("[Info] Robot control is always active in record mode.")
        else:
            print("[B] Start robot control")
            print("[R] Stop robot control and reset")

    def _on_press(self, key) -> None:
        key_char = getattr(key, "char", None)
        if key_char is None:
            return
        key_char = key_char.lower()
        if self.mode == "record":
            if key_char == "b":
                self._start_recording()
            elif key_char == "n":
                self._save_episode()
            elif key_char == "r":
                self._skip_episode()
        elif key_char == "b":
            self._started = True
            self._reset_state = False
        elif key_char == "r":
            self._started = False
            self._reset_state = True
            self._call_callback("R")

    def _on_joystick_trigger(self, msg) -> None:
        if self.mode != "record" or msg is None:
            return
        trigger = getattr(msg, "data", "")
        if trigger == "right":
            if self._episode_phase == "recording":
                self._save_episode()
            else:
                self._start_recording()
        elif trigger == "left":
            self._skip_episode()

    def _call_callback(self, key: str) -> None:
        callback = self._additional_callbacks.get(key)
        if callback is not None:
            callback()

    def _start_recording(self) -> None:
        if self._episode_phase == "recording":
            return
        print("[Control] Start recording requested.")
        self._started = True
        self._reset_state = False
        self._episode_phase = "recording"
        self._call_callback("B")

    def start_recording(self) -> None:
        self._start_recording()

    def _save_episode(self) -> None:
        if self.mode == "record" and self._episode_phase != "recording":
            print("[Control] Save ignored because recording has not started.")
            return
        print("[Control] Save episode requested.")
        self._started = False
        self._reset_state = True
        self.clear_command_cache()
        self._call_callback("N")
        self._episode_phase = "idle"

    def _skip_episode(self) -> None:
        print("[Control] Reset/skip episode requested.")
        self._started = False
        self._reset_state = True
        self.clear_command_cache()
        self._call_callback("R")
        self._episode_phase = "idle"

    def get_action(self):
        if self._reset_state:
            self._reset_state = False
            return {"reset": True}
        if self.mode == "inference" and not self._started:
            return None
        return super().get_action()

    def reset(self) -> None:
        self._reset_state = False
        super().reset()

    def add_callback(self, key: str, func: Callable) -> None:
        self._additional_callbacks[key] = func

    def shutdown(self) -> None:
        if self.listener is not None:
            try:
                self.listener.stop()
            except Exception:
                pass
            self.listener = None
        self.close()
        print("FFWSG2Sdk shutdown complete")
