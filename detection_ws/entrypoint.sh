#!/usr/bin/bash

source /opt/ros/humble/setup.sh
source /root/detection_ws/install/setup.sh

ros2 launch mock_detection mock_detection.launch.py
