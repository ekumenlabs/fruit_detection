from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description() -> LaunchDescription:
    """Launch description to initiate the detection node.
    
    ----
    Returns:
        LaunchDescription: With just the detection_node.
    """
    ld = LaunchDescription()
    detection_node = Node(
        package="detection",
        executable="detection_node",
        parameters=[
            {'model_path': '/root/detection_ws/model/model.pth'}
        ]
    )
    ld.add_action(detection_node)
    return ld