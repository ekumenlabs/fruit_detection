import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class MockDetectionNode(Node):
    """Emulates a detection by reading images from /image_raw and publishing a smoothed
    version into /proc_image. On top of that, generates a mock detection and publishes
    that to /detection. 
    """

    TARGET_ENCODING="bgr8"
    KERNEL_SIZE_PXL=(11,11)
    KERNEL_NORM=KERNEL_SIZE_PXL[0]**2
    TOPIC_QOS_QUEUE_LENGTH=10

    def __init__(self) -> None:
        """Constructor"""
        super().__init__('mock_detection_node')

        self.image_subscription = self.create_subscription(Image, "/image_raw", self.image_callback, MockDetectionNode.TOPIC_QOS_QUEUE_LENGTH)
        self.image_publisher = self.create_publisher(Image, '/proc_image', MockDetectionNode.TOPIC_QOS_QUEUE_LENGTH)
        self.cv_bridge = CvBridge()
        self.smoothing_kernel = np.ones(MockDetectionNode.KERNEL_SIZE_PXL,np.float32)/MockDetectionNode.KERNEL_NORM
   
    def image_callback(self, msg: Image) -> None:
        """Image callback.
        Produces the detection output together with the smoothed image.

        Args:
        -----
            msg (Image): Received image.
        """
        msg_header = msg.header
        cv_frame = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding=MockDetectionNode.TARGET_ENCODING)
        cv_frame = cv2.filter2D(cv_frame,-1,self.smoothing_kernel)
        self.image_publisher.publish(
            self.cv_bridge.cv2_to_imgmsg(
                cv_frame,
                encoding=MockDetectionNode.TARGET_ENCODING,
                header=msg_header
            )
        )


def main(args=None) -> None:
    """Main function."""
    rclpy.init(args=args)
    node = MockDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()