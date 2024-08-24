import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    ObjectHypothesis,
    ObjectHypothesisWithPose,
)
from cv_bridge import CvBridge
import cv2
import numpy as np


class MockDetectionNode(Node):
    """Emulates a detection by reading images from /image_raw and publishing another image
    into /proc_image with the detections. On top of that, generates a mock detection and publishes
    that to /detections. 
    """

    TARGET_ENCODING="bgr8"
    TOPIC_QOS_QUEUE_LENGTH=10
    RECT_COLOR=(0,255,0)
    # TODO: unhardcode path.
    CLASSIFIER_CONFIG="/root/detection_ws/install/mock_detection/share/mock_detection/config/haarcascade_frontalface_defaults.xml"

    def __init__(self) -> None:
        """Constructor"""
        super().__init__('mock_detection_node')

        self.image_subscription = self.create_subscription(Image, "/image_raw", self.image_callback, MockDetectionNode.TOPIC_QOS_QUEUE_LENGTH)
        self.image_publisher = self.create_publisher(Image, '/proc_image', MockDetectionNode.TOPIC_QOS_QUEUE_LENGTH)
        self.detections_publisher = self.create_publisher(Detection2DArray, '/detections', MockDetectionNode.TOPIC_QOS_QUEUE_LENGTH)
        self.cv_bridge = CvBridge()
        self.classifier = cv2.CascadeClassifier(MockDetectionNode.CLASSIFIER_CONFIG)
   
    def image_callback(self, msg: Image) -> None:
        """Image callback.
        Produces the detection output together with the smoothed image.

        Args:
        -----
            msg (Image): Received image.
        """
        msg_header = msg.header
        cv_frame = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding=MockDetectionNode.TARGET_ENCODING)
        detections = self.compute_detections(cv_frame)
        self.draw_rects(cv_frame, detections, MockDetectionNode.RECT_COLOR)
        detections_msg = self.rects_to_ros2_detections(detections, msg_header)

        self.image_publisher.publish(
            self.cv_bridge.cv2_to_imgmsg(
                cv_frame,
                encoding=MockDetectionNode.TARGET_ENCODING,
                header=msg_header
            )
        )
        self.detections_publisher.publish(detections_msg)

    def compute_detections(self, cv_img):
        """Computes face detections.
        
        Converts the input image in a gray scale image, to then apply the classiffier
        and return a list of rectangles with the detections.
        
        Args:
        -----
            cv_img (cv2.Mat): The input image to apply the classifier.
        
        Returns:
        --------
            list[tuple[int, int, int, int]]: The list of rectangles with the detections.
        """
        cv_gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        detections = self.classifier.detectMultiScale(cv_gray_img, 1.3, 4)
        if len(detections) == 0:
            return []
        detections[:,2:] += detections[:,:2]
        return detections

    def draw_rects(self, cv_img, rects, color) -> None:
        """Draws rectangles in the provided image.
        
        Args:
        -----
            cv_img (cv2.Mat): The input image to apply the rectangles.
            rects (list[tuple[int, int, int, int]]): The list of rectangles to draw.
            color (list[tuple[int, int, int]]): The color of each rectangle border.
        """
        for x1, y1, x2, y2 in rects:
            cv2.rectangle(cv_img, (x1, y1), (x2, y2), color, 2)
    
    def rects_to_ros2_detections(self, rects, header):
        """Creates a detection result as if there where multiple classes.

        Notes: there are multiple values hardcoded. It is expected to evolve the method into
        something that relates to the actual classes in the future. Also, the score in the
        hyphotesis.
        Provided that there is no pose estimation, only the bounding box is given.

        Args:
        -----
            rects (list[tuple[int, int, int, int]]): The list of detections to include.
            header (Header): The header stamp of the message to return.
        
        Returns:
        --------
            Detection2DArray: The detection array with the mulitple detections when found.
        """
        result = Detection2DArray()
        result.header = header
        result.detections = []
        for x1, y1, x2, y2 in rects:
            detection_2d = Detection2D()
            detection_2d.header = header
            bbox = BoundingBox2D()
            bbox.size_x = float(x2 - x1)
            bbox.size_y = float(y2 - y1)
            bbox.center.theta = 0.
            bbox.center.position.y = (y2 - y1) / 2
            bbox.center.position.y = (y2 - y1) / 2
            detection_2d.bbox = bbox
            detection_2d.id = "face"
            object_hypothesis_with_pose = ObjectHypothesisWithPose()
            object_hypothesis = ObjectHypothesis()
            object_hypothesis.class_id = "face"
            object_hypothesis.score = 0.8
            object_hypothesis_with_pose.hypothesis = object_hypothesis
            detection_2d.results.append(object_hypothesis_with_pose)
            result.detections.append(detection_2d)
        return result


def main(args=None) -> None:
    """Main function."""
    rclpy.init(args=args)
    node = MockDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()