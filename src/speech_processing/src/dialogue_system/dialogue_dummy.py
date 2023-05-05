

class dialogue_system: 
    def __init__(self):
        self.robot_state = None
        self.robot_interruptable= False
        
    def process_speech_input(self,text,command, age,confidence):
        if command == "stop" and confidence>0.5:

            return ("stop",age, confidence,"None","Red","Günther","Kitchen")
        
    def process_commandline_input(self,command):
        
        if command == "stop":
            if self.robot_interruptable == False:
                print("robot not interruptable")
                return "None"
            else:
                return "stop"
        elif command == "continue":
            if self.robot_state=="working":
                print("already working")
                return "None"
            else: 
                return "continue"


    def process_robot_input(self,robot_state,robot_interruptable):
        self.robot_state = robot_state
        self.robot_interruptable = robot_interruptable
