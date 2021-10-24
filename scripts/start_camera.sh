#!/bin/bash

# setup ros environment
source "/opt/ros/$ROS_DISTRO/setup.bash"
echo "Starting ros"
roscore &
echo "Waiting for ros to start"
sleep 5
echo "Launch camera node"
rosparam set cv_camera/device_path /dev/sensors/camera
rosrun cv_camera cv_camera_node &

echo "Launch streamer"
./scripts/stream.py
# This sets line buffering of stdout and stderr so that ROS output does not
# get buffered
stdbuf -oL -eL $@

