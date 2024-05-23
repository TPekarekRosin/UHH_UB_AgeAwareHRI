# UHH_UB_AgeAwareHRI
Collaboration between UHH and UB for age aware HRI.

### Installation
*setup.bash* installs the necessary python packages in a virtual environment *~/.arvenv*.

The Git Repository is treated as a catkin workspace. 

*setup.bash* also handles the installation of ROS requirements.

Requires **Python >= 3.8**.

1) Make shell file executable with
    ```bash
   chmod +x setup.bash
    ```
3) Then execute with
    ```bash
   ./setup.bash
    ```
5) Then initialize pycram with
    ```bash
   ./setup_pycram.bash
    ```
During installation console will ask for sudo password.

### IF PYCRAM WAS UPDATED

1) Clean
    ```bash
   ./clean_pycram.sh 
   ``` 
2) If the pycram branch was updated, pull all changes using
   ```bash
   ./update_pycram.sh
    ```
3) install
   ```bash
   ./setup_pycram.bash
    ```
During installation console will ask for sudo password.


### Start-up
1) Open a terminal and start ROS with ```roscore```.
2) Navigate to UHH_UB_AgeAwareHRI Git repository and source the catkin workspace: (this has do be done for all terminals you are opening)
   ```bash
   source ~/.arvenv/bin/activate && source ./devel/setup.bash
   ```
4) Start these in separate terminals. 
   Start the speech and age recognition:
    ```bash
   rosrun speech_processing speech_processing_client.py
   ```
   Start the dialogue manager:
   (Tips: In this work, we use ChatGPT as the dialog manager, therefore you need an OpenAI key to run the code. To use OpenAI key, you need to create a file named "openai_api_key.txt" in our project folder and paste the key into this file.)
    ```bash
   rosrun speech_processing dialogue_system_node.py
   ```
   Start the speech synthesis:
    ```bash
   rosrun speech_processing speech_synthesis.py
   ```
 
5) To start the pycram, first make sure you initialized pycram as outlined in 3) of the installation. Open another terminal and source the workspace:
   Navigate to UHH_UB_AgeAwareHRI Git repository and source the catkin workspace: (this has do be done for all terminals you are opening)
   ```bash
   source ~/.arvenv_pycram/bin/activate && source ./devel/setup.bash
   ```
  
   Manually make the demo file and the robot bash script executable:
   ```
   chmod +x run_robot_demo.sh && chmod +x src/cognitive_archi/pycram/demos/frontiers/interrupt_demo.py
   ```
   Start the pycram demo using (Note: this will launch rviz with an appropriate config, and the demo will start as soon as everything has loaded):
   ```bash
   ./run_robot_demo.sh 
   ```
  
   
### Instructions to interrupting the PR2's actions per hand (testing per hand)
Open up a new terminal in your workspace, and source the workspace using ```./setup.bash```
The Interrupt Client can be tested injecting the following rostopic pub command at various points in the demo, whenever the robot prints "I am now interruptable for 5 seconds":
   ```bash
   rostopic pub /robot_minor_interruption speech_processing/message_to_robot "command: ''
   age: 0
   confidence: 0.0
   add_object:
   - {type: 'cereal', color: 'red', name: '', location: '', size: ''}
   del_object:
   - {type: 'milk', color: 'blue', name: '', location: '', size: ''}" 
   ``` 
   
If you want to switch from cereal back to the milk, after interrupting once use:
   ```bash
   rostopic pub /robot_minor_interruption speech_processing/message_to_robot "command: ''
   age: 0
   confidence: 0.0
   add_object:
   - {type: 'milk', color: 'blue', name: '', location: '', size: ''}
   del_object:
   - {type: 'cereal', color: 'red', name: '', location: '', size: ''}"
   ```
 
  
   
   
   
   
