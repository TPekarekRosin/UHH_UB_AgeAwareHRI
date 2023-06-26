import rospy


class dialogue_system: 
    def __init__(self):
        self.robot_state = None
        self.robot_interruptable = False
        
        self.objects_in_use = []
        
    def process_speech_input(self, transcript, age, confidence):
        # dummy code
        if transcript == "stop" and confidence > 0.5:
            return 'stop', age, confidence, ['', '', '', '', ''], ['', '', '', '', '']
        
        elif 'coffee' in transcript and confidence > 0.5:
            return 'coffee', age, confidence, ['cup', '', '', '', 'big'], ['', '', '', '', '']
        
    def process_commandline_input(self, command):
        # dummy code
        if command == "stop":
            if not self.robot_interruptable:
                print("robot not interruptable")
                return "do_not_send", "None"
            else:
                return "major", ('stop', 60, 0.9, ['', '', '', '', ''], ['', '', '', '', ''])
        elif command == "continue":
            if self.robot_state == "working":
                print("already working")
                return "minor", "None"
            else: 
                return "minor", ('continue', 60, 0.9, ['', '', '', '', ''], ['', '', '', '', ''])
    
    def get_objects_in_use(self):
        # looks at state of rosparam \objects_in_use and return list of objects
        param_name = rospy.search_param('object_in_use')
        object_str = rospy.get_param(param_name)
        self.objects_in_use = object_str.split(',')

    def process_robot_input(self, state, interruptable, object_info,
                            move_arm, move_base, current_loc, destination_loc):
        self.robot_state = state
        self.robot_interruptable = interruptable
        string_for_synthesizer = f"I am {state}"
        return string_for_synthesizer
    
    def get_robot_states(self):
        return self.robot_state, self.robot_interruptable