"""Shared ROS clock for bringup state and sensor topics."""

from cyclo_lab.runtime.transport.ros2_zenoh import CLOCK, create_publisher, now_time_msg, reliable_qos


class ClockPublisher:
    """Publish Unix wall time once per control step, then reuse it in message headers.

    This intentionally follows the existing live-robot topic time base, not
    elapsed simulation time. Pausing or resetting the environment does not reset it.
    """

    def __init__(self) -> None:
        self.writer = create_publisher("/clock", CLOCK, qos=reliable_qos(10))

    @property
    def endpoints(self) -> tuple:
        return (self.writer,)

    def publish(self):
        stamp = now_time_msg()
        self.writer.publish(clock=stamp)
        return stamp
