# UHH_UB_AgeAwareHRI
This framework offers an adaptive approach to human-robot interaction. We use age as a modulating variable for the interaction and allow the user to interrupt and change the robot's actions during the interaction itself.

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

1) If the pycram branch was updated, pull all changes using
   ```bash
   ./update_pycram.sh
    ```
   ```bash
   ./setup_pycram.bash
    ```
   If the script fails because there are uncommited changes, please make sure to
   restore any changes not staged for commit inside the pycram repository.
   Please note that this will only occur if you make changes to the pycram code on your side
   
During installation console may ask for sudo password.


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

   Start the pycram demo using (Note: this will launch rviz with an appropriate config, and the demo will start as soon as everything has loaded):
   ```bash
   ./run_robot_demo.sh 
   ```

6) Below is a list of objects and their attributes that are currently available.
   
   Two milks:
   1. milk1 has type "milk", color "blue", size "Normal"
   2. milk2 has type "milk", color "red", size "Big"

   One Bowl:
   1. bowl has type "bowl", color "white", size "Normal"

   One Cereal:
   1. cereal has type "cereal", color "green", size "Normal"
  
   
### Instructions to interrupting the PR2's actions per hand (testing per hand)
Open up a new terminal in your workspace, and source the workspace using ```./setup.bash```
The Interrupt Client can be tested injecting the following rostopic pub command at various points in the demo, whenever the robot prints "I am now interruptable for *X* seconds":
   ```bash
   rostopic pub /robot_minor_interruption speech_processing/message_to_robot "command: 'bring_me'
   age: 0
   confidence: 0.0
   add_object:
   - {type: 'cereal', color: 'green', name: '', location: '', size: 'Normal'}
   del_object:
   - {type: 'milk', color: 'red', name: '', location: '', size: 'Big'}" 
   ``` 
   
If you want to switch from cereal back to the milk, after interrupting once use:
   ```bash
   rostopic pub /robot_minor_interruption speech_processing/message_to_robot "command: 'bring_me'
   age: 0
   confidence: 0.0
   add_object:
   - {type: 'milk', color: 'red', name: '', location: '', size: 'Big'}
   del_object:
   - {type: 'cereal', color: 'green', name: '', location: '', size: 'Normal'}"
   ```
If you want to change location:
```bash
rostopic pub /robot_minor_interruption speech_processing/message_to_robot "command: 'change_location'
age: 0
confidence: 0.0
add_object:

{type: '', color: '', name: '', location: 'table', size: ''}
del_object:
{type: '', color: '', name: '', location: '', size: ''}"
```

## Possible Installation Errors
### Compatibility between CUDA version and faster-whisper
- Solution 1: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/{path to home repository}/.arvenv/lib/python3.8/site-packages/nvidia/cudnn/lib
- Solution 2: Downgrade faster-whisper to version 0.10.1 for CUDA
    ```bash
    pip uninstall faster-whisper
    pip install faster-whisper==0.10.1
    pip install transformers -U
    ```

 
## Appreciation
- [Faster Whisper](https://github.com/SYSTRAN/faster-whisper/tree/master): ASR model speed-up.
- [Live Wav2Vec2](https://github.com/oliverguhr/wav2vec2-live/tree/main): Inspiration for the setup with multi-thread VAD and ASR for Transformers.
- [Silero VAD](https://github.com/snakers4/silero-vad): Fast and reliable VAD, we specifically used version 4.0 for processing of larger audio-chunks.

   
   
   
   
