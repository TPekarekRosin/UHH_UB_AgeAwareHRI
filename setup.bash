#!/bin/bash

# parameters
: ${PYTHON="/usr/bin/python3"} # python executable of virtualenv
: ${VIRTUALENVDIR="arvenv"} # directory name of virtualenv in home (with a '.' prefix)
# get dir
CALLDIR=`pwd`
cd "`dirname "$BASH_SOURCE"`"
WORKDIR=$(pwd)
cd "$CALLDIR"
VIRTUALENV="virtualenv"
echo Running at: "$WORKDIR"

#virtualenv setup
echo "Checking for virtualenv"
if [ -d ".$VIRTUALENVDIR/" ]; then
  echo "Existing virtualenv found"
else
  echo "No virtualenv found - setting up new virtualenv"
  # Test for virtualenv
  if ! [ -x "$(command -v $VIRTUALENV)" ]; then
    echo -e "\e[31mERROR: Could not find command $VIRTUALENV, please make sure it is installed or change the value of VIRTUALENV to the proper command.\e[0m"
  fi
  $VIRTUALENV -p $PYTHON ~/.$VIRTUALENVDIR
fi
echo "Activating virtualenv"
source ~/.$VIRTUALENVDIR/bin/activate

#install python packages
pip install wget
sudo apt-get install sox libsndfile1 ffmpeg  portaudio19-dev swig
pip install -r requirements.txt

# source setup
source /opt/ros/noetic/setup.bash

# build catkin workspace
echo "Building ROS packages"
cd $WORKDIR

if [ -x "$(command -v catkin_make)" ]; then
  sudo apt-get install cmake python3-catkin-pkg python3-rospy python-nose python-setuptools libgtest-dev build-essential
  pip install rospkg empy
  catkin_make -DPYTHON_EXECUTABLE=~/.$VIRTUALENVDIR/bin/python3
  source $WORKDIR/devel/setup.bash
fi
if ! [ -x "$(command -v catkin_make)" ]; then
  echo "Catkin not found - skipping ROS packages"
fi
