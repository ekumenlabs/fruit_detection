"""Launch the fruit detection node."""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Launch description to initiate the detection node.

    ----
    Returns:
        LaunchDescription: With just the fruit_detection_node.
    """
    ld = LaunchDescription()
    fruit_detection_node = Node(
        package="fruit_detection",
        executable="fruit_detection_node",
        parameters=[{"model_path": "/root/detection_ws/model/model.pth"}],
    )
    ld.add_action(fruit_detection_node)
    return ld
