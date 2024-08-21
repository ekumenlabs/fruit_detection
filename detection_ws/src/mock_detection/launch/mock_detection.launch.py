from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description() -> LaunchDescription:
    """Launch description to initiate the mock_detection node.
    
    ----
    Returns:
        LaunchDescription: With just the mock_detection_node.
    """
    ld = LaunchDescription()
    mock_detection_node = Node(
        package="mock_detection",
        executable="mock_detection_node",
    )
    ld.add_action(mock_detection_node)
    return ld