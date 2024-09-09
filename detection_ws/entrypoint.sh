#!/usr/bin/bash

source /opt/ros/humble/setup.sh
source /root/detection_ws/install/setup.sh

ros2 launch detection detection.launch.py
