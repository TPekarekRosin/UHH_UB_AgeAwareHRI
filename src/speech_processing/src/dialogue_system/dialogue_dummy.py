import rospy

class dialogue_system: 
    def __init__(self):
        self.robot_state = None
        self.robot_interruptable= False
        
    def process_speech_input(self,text,command, age,confidence):
        if command == "stop" and confidence>0.5:

            return ("stop",age, confidence,"None","None","Red","Günther","Kitchen","large")
        if command == "can i have coffee" and confidence>0.5 :
            return ("coffee", age, confidence, "mug", "mug", "red", "name","counter","large")
        
    def process_commandline_input(self,command):
        
        if command == "stop":
            if self.robot_interruptable == False:
                print("robot not interruptable")
                return "do_not_send","None"
            else:
                return "major" ,("stop", 60, 1, "mug", "mug", "red", "name","counter","large")
        elif command == "continue":
            if self.robot_state=="working":
                print("already working")
                return "minor", "None"
            else: 
                return "minor",("continue", 60, 1, "mug", "mug", "red", "name","counter","large")
                
                #return "continue"
    
    def get_objects_in_use(self):
        # looks at state of rosparam \objects_in_use and return list of objects
        param_name = rospy.search_param('object_in_use')
        object_str = rospy.get_param(param_name)
        return object_str.split(',')

    def process_robot_input(self,robot_state,robot_interruptable):
        self.robot_state = robot_state
        self.robot_interruptable = robot_interruptable
        string_for_synthesizer = f"I am {robot_state}"
        return string_for_synthesizer
    def get_robot_states(self):
        return self.robot_state , self.robot_interruptable 