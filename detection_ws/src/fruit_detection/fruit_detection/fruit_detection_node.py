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
import torch
import torchvision
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

_FRUIT_CATEGORIES={
    0: "background",
    1: "apple",
    2: "avocado",
    3: "lime",
}

def get_transform():
    transforms = []
    transforms.append(T.ToPILImage())
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    return T.Compose(transforms)

class FruitDetectionNode(Node):
    """
    Reads images from /image_raw and publishes another image
    into /proc_image with detections made with a fasterrcnn_resnet50_fpn model.
    On top of that, it publishes the bounding box of the detections to /detections. 
    """

    TARGET_ENCODING="bgr8"
    TOPIC_QOS_QUEUE_LENGTH=10
    RECT_COLOR=(0, 0, 255)
    SCORE_THRESHOLD=0.7

    def __init__(self) -> None:
        """Constructor"""
        super().__init__('detection_node')
        self.declare_parameter('model_path', 'model.pth')
        self.__model_path = self.get_parameter('model_path').get_parameter_value().string_value

        self.image_subscription = self.create_subscription(Image, "/image_raw", self.image_callback, FruitDetectionNode.TOPIC_QOS_QUEUE_LENGTH)
        self.image_publisher = self.create_publisher(Image, '/proc_image', FruitDetectionNode.TOPIC_QOS_QUEUE_LENGTH)
        self.detections_publisher = self.create_publisher(Detection2DArray, '/detections', FruitDetectionNode.TOPIC_QOS_QUEUE_LENGTH)
        self.cv_bridge = CvBridge()
        self.device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device_str)
        self.load_model()
        self.get_logger().info(f" device? {self.device_str} version {torch.__version__}")
        self.ingest_transform = get_transform()

    def load_model(self):
        """Load the torch model"""
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(_FRUIT_CATEGORIES))
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load(self.__model_path, weights_only=True))
        self.model.eval()
        self._labels = _FRUIT_CATEGORIES

    def image_to_tensor(self, img):
        transformed_img = self.ingest_transform(img)
        return [transformed_img.to(self.device)]

    def score_frame(self, frame):
        """
        function scores each frame of the video and returns results.
        :param frame: frame to be infered.
        :return: labels and coordinates of objects found. 
        """
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            output = self.model(self.image_to_tensor(img))
            # As we only feed one image, we should get only one output
            output = output[0]
            results = []
            for i, (box, score, label) in enumerate(zip(output['boxes'], output['scores'], output['labels'])):
                if score >= FruitDetectionNode.SCORE_THRESHOLD:
                    results.append({'box':box,'score':score, 'label':self._labels[label.item()]})
        return results
   
    def plot_boxes(self, detections, frame):
        """
        Plots boxes and labels on frame.
        :param detections: inferences made by model
        :param frame: frame on which to  make the plots
        :return: new frame with boxes and labels plotted.
        """
        for detection in detections:
            row = detection['box']
            x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            cv2.rectangle(frame, (x1, y1), (x2, y2), FruitDetectionNode.RECT_COLOR, 1)
            cv2.putText(frame, str(detection['label']), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    def detection_to_ros2(self, detections, header):
        """
        Creates a detection result as if there where multiple classes.

        Notes: there are multiple values hardcoded. It is expected to evolve the method into
        something that relates to the actual classes in the future. Also, the score in the
        hyphotesis.
        Provided that there is no pose estimation, only the bounding box is given.

        Args:
        -----
            detections (list[dict]): The list of detections to include.
            header (Header): The header stamp of the message to return.
        
        Returns:
        --------
            Detection2DArray: The detection array with the mulitple detections when found.
        """
        result = Detection2DArray()
        result.header = header
        result.detections = []
        for detection in detections:
            x1, y1, x2, y2 = detection['box']
            detection_2d = Detection2D()
            detection_2d.header = header
            bbox = BoundingBox2D()
            bbox.size_x = float(x2 - x1)
            bbox.size_y = float(y2 - y1)
            bbox.center.theta = 0.
            bbox.center.position.y = float(y2 - y1) / 2.0
            bbox.center.position.y = float(y2 - y1) / 2.0
            detection_2d.bbox = bbox
            detection_2d.id = detection["label"]
            object_hypothesis_with_pose = ObjectHypothesisWithPose()
            object_hypothesis = ObjectHypothesis()
            object_hypothesis.class_id = detection['label']
            object_hypothesis.score = float(detection['score'])
            object_hypothesis_with_pose.hypothesis = object_hypothesis
            detection_2d.results.append(object_hypothesis_with_pose)
            result.detections.append(detection_2d)
        return result
    
    def image_callback(self, msg: Image) -> None:
        """Image callback.
        Produces the detection output together with the smoothed image.

        Args:
        -----
            msg (Image): Received image.
        """
        msg_header = msg.header
        cv_frame = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding=FruitDetectionNode.TARGET_ENCODING)
        detections = self.score_frame(cv_frame)
        self.plot_boxes(detections, cv_frame)
        detections_msg = self.detection_to_ros2(detections, msg_header)

        self.image_publisher.publish(
            self.cv_bridge.cv2_to_imgmsg(
                cv_frame,
                encoding=FruitDetectionNode.TARGET_ENCODING,
                header=msg_header
            )
        )
        self.detections_publisher.publish(detections_msg)


def main(args=None) -> None:
    """Main function."""
    rclpy.init(args=args)
    node = FruitDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
