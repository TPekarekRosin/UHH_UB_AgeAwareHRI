#!/bin/bash

# parameters
: ${PYTHON="/usr/bin/python3"} # python executable of virtualenv
: ${VIRTUALENVDIR="arvenv_pycram"} # directory name of virtualenv in home (with a '.' prefix)
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

git submodule init
git submodule update
cd src
vcs import --input https://raw.githubusercontent.com/sunava/pycram/suturo/pycram.rosinstall --recursive

#install python packages
pip install -r pycram/requirements.txt

rosdep install --from-paths pycram --ignore-src -r -y

# source setup
source /opt/ros/noetic/setup.bash

# build catkin workspace
echo "Building ROS packages"
cd $WORKDIR
git submodule init
git submodule update

PACKAGE_DIR=$WORKDIR/src/pycram  # Package directory
LAUNCH_DIR=$PACKAGE_DIR/launch  # Launch directory
PYTHON_SCRIPT_PATH=$PACKAGE_DIR/demos/frontiers/interrupt_demo.py  # Path to your Python script
LAUNCH_FILE_NAME=run_interrupt_demo.launch  # Name of your launch file
CONFIG_FILE_NAME=pycram_rviz_config.rviz

# Create the launch file
cat <<EOF >$LAUNCH_DIR/$LAUNCH_FILE_NAME
<launch>

    <include file="$PACKAGE_DIR/launch/ik_and_description.launch"/>
    
    <node name="rviz" pkg="rviz" type="rviz" args="-d $PACKAGE_DIR/launch/pycram_rviz_config.rviz" required="true"/>
    
    <node name="interrupt_demo_node"
          pkg="pycram"
          type="interrupt_demo.py"
          output="screen">
    </node>
</launch>
EOF

cat <<EOF >$LAUNCH_DIR/$CONFIG_FILE_NAME
Panels:
  - Class: rviz/Displays
    Help Height: 78
    Name: Displays
    Property Tree Widget:
      Expanded:
        - /Global Options1
        - /Status1
        - /MarkerArray1
      Splitter Ratio: 0.5
    Tree Height: 549
  - Class: rviz/Selection
    Name: Selection
  - Class: rviz/Tool Properties
    Expanded:
      - /2D Pose Estimate1
      - /2D Nav Goal1
      - /Publish Point1
    Name: Tool Properties
    Splitter Ratio: 0.5886790156364441
  - Class: rviz/Views
    Expanded:
      - /Current View1
    Name: Views
    Splitter Ratio: 0.5
  - Class: rviz/Time
    Name: Time
    SyncMode: 0
    SyncSource: ""
Preferences:
  PromptSaveOnExit: true
Toolbars:
  toolButtonStyle: 2
Visualization Manager:
  Class: ""
  Displays:
    - Alpha: 0.5
      Cell Size: 1
      Class: rviz/Grid
      Color: 160; 160; 164
      Enabled: true
      Line Style:
        Line Width: 0.029999999329447746
        Value: Lines
      Name: Grid
      Normal Cell Count: 0
      Offset:
        X: 0
        Y: 0
        Z: 0
      Plane: XY
      Plane Cell Count: 10
      Reference Frame: <Fixed Frame>
      Value: true
    - Class: rviz/MarkerArray
      Enabled: true
      Marker Topic: /pycram/viz_marker
      Name: MarkerArray
      Namespaces:
        apartment: true
        cereal: true
        milk: true
        pr2: true
      Queue Size: 100
      Value: true
  Enabled: true
  Global Options:
    Background Color: 48; 48; 48
    Default Light: true
    Fixed Frame: map
    Frame Rate: 30
  Name: root
  Tools:
    - Class: rviz/Interact
      Hide Inactive Objects: true
    - Class: rviz/MoveCamera
    - Class: rviz/Select
    - Class: rviz/FocusCamera
    - Class: rviz/Measure
    - Class: rviz/SetInitialPose
      Theta std deviation: 0.2617993950843811
      Topic: /initialpose
      X std deviation: 0.5
      Y std deviation: 0.5
    - Class: rviz/SetGoal
      Topic: /move_base_simple/goal
    - Class: rviz/PublishPoint
      Single click: true
      Topic: /clicked_point
  Value: true
  Views:
    Current:
      Class: rviz/Orbit
      Distance: 7.358067989349365
      Enable Stereo Rendering:
        Stereo Eye Separation: 0.05999999865889549
        Stereo Focal Distance: 1
        Swap Stereo Eyes: false
        Value: false
      Field of View: 0.7853981852531433
      Focal Point:
        X: 2.786846399307251
        Y: 2.7975056171417236
        Z: 0.0522986464202404
      Focal Shape Fixed Size: true
      Focal Shape Size: 0.05000000074505806
      Invert Z Axis: false
      Name: Current View
      Near Clip Distance: 0.009999999776482582
      Pitch: 0.8503977656364441
      Target Frame: <Fixed Frame>
      Yaw: 6.043579578399658
    Saved: ~
Window Geometry:
  Displays:
    collapsed: true
  Height: 846
  Hide Left Dock: true
  Hide Right Dock: true
  QMainWindow State: 000000ff00000000fd000000040000000000000156000002b0fc0200000008fb0000001200530065006c0065006300740069006f006e00000001e10000009b0000005c00fffffffb0000001e0054006f006f006c002000500072006f007000650072007400690065007302000001ed000001df00000185000000a3fb000000120056006900650077007300200054006f006f02000001df000002110000018500000122fb000000200054006f006f006c002000500072006f0070006500720074006900650073003203000002880000011d000002210000017afb000000100044006900730070006c006100790073000000003d000002b0000000c900fffffffb0000002000730065006c0065006300740069006f006e00200062007500660066006500720200000138000000aa0000023a00000294fb00000014005700690064006500530074006500720065006f02000000e6000000d2000003ee0000030bfb0000000c004b0069006e0065006300740200000186000001060000030c00000261000000010000010f000002b0fc0200000003fb0000001e0054006f006f006c002000500072006f00700065007200740069006500730100000041000000780000000000000000fb0000000a00560069006500770073000000003d000002b0000000a400fffffffb0000001200530065006c0065006300740069006f006e010000025a000000b200000000000000000000000200000490000000a9fc0100000001fb0000000a00560069006500770073030000004e00000080000002e10000019700000003000004b00000003efc0100000002fb0000000800540069006d00650100000000000004b0000003bc00fffffffb0000000800540069006d00650100000000000004500000000000000000000004b0000002b000000004000000040000000800000008fc0000000100000002000000010000000a0054006f006f006c00730100000000ffffffff0000000000000000
  Selection:
    collapsed: false
  Time:
    collapsed: false
  Tool Properties:
    collapsed: false
  Views:
    collapsed: true
  Width: 1200
  X: 72
  Y: 27
EOF

cat <<EOF >$WORKDIR/run_robot_demo.sh
#!/bin/bash

source /home/me/.arvenv_pycram/bin/activate

source /opt/ros/noetic/setup.bash

cd $WORKDIR

source devel/setup.bash

roslaunch pycram run_interrupt_demo.launch

deactivate
EOF

cat <<EOF >$WORKDIR/clean_pycram.sh
#!/bin/bash

cd $PACKAGE_DIR

git restore "$PYTHON_SCRIPT_PATH"

rm $LAUNCH_DIR/$LAUNCH_FILE_NAME
rm $LAUNCH_DIR/$CONFIG_FILE_NAME

EOF

cd $WORKDIR

sudo chmod +x clean_pycram.sh

TEMP_FILE=$(mktemp)

echo "#!/usr/bin/env python3" > "$TEMP_FILE"
cat "$PYTHON_SCRIPT_PATH" >> "$TEMP_FILE"

mv "$TEMP_FILE" "$PYTHON_SCRIPT_PATH"

sudo chmod +x src/pycram/demos/frontiers/interrupt_demo.py
sudo chmod +x run_robot_demo.sh

source ~/.arvenv/bin/activate

sudo apt-get install python3-catkin-tools
catkin clean -y && catkin build
source $WORKDIR/devel/setup.bash

