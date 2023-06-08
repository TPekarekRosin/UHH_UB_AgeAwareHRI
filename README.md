# UHH_UB_AgeAwareHRI
Collaboration between UHH and UB for age aware HRI.

### Installation
*setup.bash* installs the necessary python packages in a virtual environment *~/.arvenv*.

The Git Repository is treated as a catkin workspace. 

*setup.bash* also handles the installation of ROS requirements.

Requires **Python >= 3.8**.

1) Make shell file executable with ```chmod +x setup.bash```
2) Then execute with ```./setup.bash```

During installation console will ask for sudo password.


### Start-up
1) Open a terminal and start ROS with ```roscore```.
2) Open another terminal and activate the virtual environment:
   ```bash
   source ~/.arvenv/bin/activate
   ```
3) Navigate to UHH_UB_AgeAwareHRI Git repository and source the catkin workspace:
   ```bash
    source ./devel/setup.bash
   ```
4) Start these in separate terminals. 
   Start the speech and age recognition:
    ```bash
   rosrun speech_processing speech_processing_client.py
   ```
   Start the dummy dialogue manager:
    ```bash
   rosrun speech_processing dialogue_system_node.py.py
   ```
   Start the dummy speech synthesis:
    ```bash
   rosrun speech_processing speech_synthesis.py
   ```
   Start the dummy robot:
    ```bash
   rosrun robot robot.py
   ```
