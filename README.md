# UHH_UB_AgeAwareHRI
Collaboration between UHH and UB for age aware HRI.

### Setup catkin workspace

1) Install catkin & dependencies for ROS Noetic:
```
sudo apt-get install ros-noetic-catkin
sudo apt-get install cmake python3-catkin-pkg python3-empy python-nose python-setuptools libgtest-dev build-essential
```
2) Source ROS: ```source /opt/ros/noetic/setup.bash```
3) Create catkin workspace: ```mkdir -p ~/catkin_ws/src```
4) Move to directory of workspace: ```cd ~/catkin_ws``` and ```catkin_make```
5) Source setup.bash: ```source devel/setup.bash```
6) Echo ROS path to check if successfully made: ```echo $ROS_PACKAGE_PATH```

