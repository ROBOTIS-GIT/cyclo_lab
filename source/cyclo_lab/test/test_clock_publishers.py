"""Clock and outgoing timestamp contract; run with Isaac Lab's Python and unittest."""

import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from cyclo_lab.runtime.bridges.sg2.topic_bridge import FFWSG2TopicBridge
from cyclo_lab.runtime.bridges.sh5.bridge import FFWSH5TopicBridge
from cyclo_lab.runtime.publishers import camera_publishers
from cyclo_lab.runtime.publishers.articulation_state_publisher import ArticulationStatePublisher
from cyclo_lab.runtime.transport.ros2_zenoh import (
    QosDurability,
    QosHistoryKind,
    QosReliability,
    time_msg,
)


class ClockPublisherTests(unittest.TestCase):
    def test_clock_type_qos_and_returned_stamp(self):
        module = importlib.import_module("cyclo_lab.runtime.publishers.clock_publisher")
        stamp = time_msg(1794896000, 123456789)
        with patch.object(module, "create_publisher") as create, patch.object(
            module, "now_time_msg", return_value=stamp
        ):
            clock = module.ClockPublisher()
            self.assertEqual(create.call_args.args, ("/clock", "rosgraph_msgs/msg/Clock"))
            qos = create.call_args.kwargs["qos"]
            self.assertEqual(qos.reliability, QosReliability.RELIABLE)
            self.assertEqual(qos.durability, QosDurability.VOLATILE)
            self.assertEqual(qos.history_kind, QosHistoryKind.KEEP_LAST)
            self.assertEqual(qos.history_depth, 10)
            self.assertIs(clock.publish(), stamp)
            create.return_value.publish.assert_called_once_with(clock=stamp)
            self.assertEqual(clock.endpoints, (create.return_value,))

    def test_sg2_reuses_clock_stamp_for_state_and_cameras(self):
        bridge = FFWSG2TopicBridge.__new__(FFWSG2TopicBridge)
        stamp = time_msg(12, 200000000)
        bridge.clock_publisher = Mock()
        bridge.clock_publisher.publish.return_value = stamp
        bridge.state_publisher = Mock()
        bridge.camera_publishers = Mock()
        bridge.publish_observations()
        bridge.clock_publisher.publish.assert_called_once_with()
        bridge.state_publisher.publish_all.assert_called_once_with(stamp=stamp)
        bridge.camera_publishers.publish.assert_called_once_with(stamp=stamp)

    def test_sh5_reuses_clock_stamp_for_state_and_cameras(self):
        bridge = FFWSH5TopicBridge.__new__(FFWSH5TopicBridge)
        stamp = time_msg(12, 200000000)
        bridge.env = SimpleNamespace(step_dt=1.0 / 15.0)
        bridge.clock_publisher = Mock()
        bridge.clock_publisher.publish.return_value = stamp
        bridge.update_odometry = Mock()
        bridge.publish_odometry = Mock()
        bridge.state_publisher = Mock()
        bridge.camera_publishers = Mock()
        bridge.publish_observations()
        bridge.clock_publisher.publish.assert_called_once_with()
        bridge.update_odometry.assert_called_once_with(bridge.env.step_dt)
        bridge.state_publisher.publish_joint_states.assert_called_once_with(stamp=stamp)
        bridge.publish_odometry.assert_called_once_with(stamp=stamp)
        bridge.state_publisher.publish_tf.assert_called_once_with(stamp=stamp)
        bridge.camera_publishers.publish.assert_called_once_with(stamp=stamp)

    def test_state_publisher_preserves_explicit_stamp_and_legacy_default(self):
        publisher = ArticulationStatePublisher.__new__(ArticulationStatePublisher)
        publisher.publish_joint_states = Mock()
        publisher.publish_odometry = Mock()
        publisher.publish_tf = Mock()
        stamp = time_msg(0, 0)
        with patch("cyclo_lab.runtime.publishers.articulation_state_publisher.now_time_msg") as now:
            publisher.publish_all(stamp=stamp)
            now.assert_not_called()
            publisher.publish_joint_states.assert_called_once_with(stamp=stamp)
            publisher.publish_odometry.assert_called_once_with(stamp=stamp)
            publisher.publish_tf.assert_called_once_with(stamp=stamp)
            publisher.publish_all()
            now.assert_called_once_with()
            publisher.publish_joint_states.assert_called_with(stamp=now.return_value)

    def test_camera_worker_preserves_capture_stamp_through_encoding(self):
        rgb = torch.zeros((1, 8, 8, 3), dtype=torch.uint8)
        camera = SimpleNamespace(data=SimpleNamespace(output={"rgb": rgb}))

        class Scene(dict):
            @property
            def sensors(self):
                return self

        stamp = time_msg(12, 999999999)
        with patch.object(camera_publishers, "create_publisher") as create, patch.object(
            camera_publishers, "now_time_msg", return_value=time_msg(20, 0)
        ) as now:
            publishers = camera_publishers.CompressedCameraPublishers(
                Scene(cam_head=camera), {"cam_head": "/test/image/compressed"}, None
            )
            try:
                publishers.publish(stamp=stamp)
                publishers._queue.join()
                now.assert_not_called()
                message = create.return_value.publish.call_args.kwargs
                self.assertIs(message["header"].stamp, stamp)
                self.assertEqual(message["format"], "jpeg")
                self.assertGreater(len(message["data"]), 0)
                publishers.publish()
                publishers._queue.join()
                now.assert_called_once_with()
                self.assertIs(create.return_value.publish.call_args.kwargs["header"].stamp, now.return_value)
            finally:
                publishers.close()
            self.assertFalse(publishers._worker.is_alive())


if __name__ == "__main__":
    unittest.main()
