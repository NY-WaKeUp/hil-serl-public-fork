"""ROS2 image subscription — same read()/close() contract as RSCapture."""

import threading
import time

import numpy as np
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class RosImageCapture:
    """Subscribe to sensor_msgs/Image; read() returns (ok, bgr uint8 ndarray)."""

    def __init__(self, name: str, topic: str, ros_node: Node, dim=(1280, 720)):
        self.name = name
        self.topic = topic
        self.dim = dim
        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._sub = ros_node.create_subscription(
            Image, topic, self._on_image, qos_profile_sensor_data
        )

    def _on_image(self, msg: Image) -> None:
        if msg.encoding in ("rgb8", "RGB8"):
            rgb = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            bgr = rgb[..., ::-1]
        else:
            bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self._lock:
            self._latest = np.ascontiguousarray(bgr)

    def read(self):
        deadline = time.time() + 5.0
        while time.time() < deadline:
            with self._lock:
                if self._latest is not None:
                    return True, self._latest.copy()
            time.sleep(0.005)
        return False, None

    def close(self) -> None:
        self._sub.destroy()
        self._sub = None
