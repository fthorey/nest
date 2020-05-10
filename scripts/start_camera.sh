#!/bin/bash

# Add local user
# Either use the LOCAL_USER_ID if passed in at runtime or
# fallback
# USER_ID=${LOCAL_USER_ID:-9001}
# echo "Starting with UID : $USER_ID"
# useradd --shell /bin/bash -u $USER_ID -o -c "" -m user
# export HOME=/home/user
# stdbuf -oL -eL $@

# setup ros environment
source "/opt/ros/$ROS_DISTRO/setup.bash"
source "/catkin_ws/devel/setup.bash"

rosparam set cv_camera/device_path /dev/sensors/camera
rosrun cv_camera cv_camera_node
# This sets line buffering of stdout and stderr so that ROS output does not
# get buffered
stdbuf -oL -eL $@

