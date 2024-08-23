#!/usr/bin/bash

source /opt/ros/humble/setup.sh

ros2 run rqt_gui rqt_gui --perspective-file /root/visualization_ws/config/image_view.perspective
