import rospy
# from speech_processing.msg import *

from langchain.chat_models import ChatOpenAI
from dialogue_system.social_brain import SocialBrain
from dialogue_system.social_env import SocialEnv

class DialogueSystem:
    def __init__(self):
        self.robot_step = None
        self.robot_interruptable = False
        
        self.objects_in_use = []
        self.user_data = dict()
        # string transcript
        # int32 age
        # float32 confidence
        self.user_data["transcript"] = "please ready to serve"
        self.user_data["age"] = 1
        self.user_data["confidence"] = 1
        self.robot_data = dict()
        with open("openai_api_key.txt") as fapi:
            self.api_key = fapi.read()
        self.env = SocialEnv()
        self.chat = ChatOpenAI(temperature=0, verbose=True, max_tokens=256, openai_api_key=self.api_key)
        self.agent = SocialBrain(model=self.chat, env=self.env)
        
    def process_speech_input(self, transcript, age, confidence):
        
        # utterance_user, 
        # age, 
        # confidence_of_age, 
        print("robot data", self.robot_data)
        
        step = self.robot_data["step"]
        interruptible = self.robot_data["interruptable"]
        dict_object = self.robot_data["object"]
        move_arm = self.robot_data["move_arm"]
        move_base = self.robot_data["move_base"]
        current_location = self.robot_data["current_location"]
        destination_location = self.robot_data["destination_location"]
        objects_in_use = self.objects_in_use
        
        # todo: define minor interruptions
        if "stop" not in transcript and confidence > 0.5:
            # todo: generate valid response for minor interruptions
            system_transcript, response_to_robot = self.agent.information_process(
                                                    transcript, age, confidence, step, interruptible, 
                                                    dict_object, move_arm, move_base, current_location, 
                                                    destination_location, objects_in_use)
            
            
            return 'minor', response_to_robot, system_transcript
        
        # todo: define major interruptions
        elif 'stop' in transcript and confidence > 0.5:
            # todo: generate valid response for major interruptions
            system_transcript = "ok i will stop anything what i am doing now"
            return 'major', (), system_transcript
        else:
            system_transcript = "i got trouble something is wrong"
            print('Command was not recognized.')
            return '', (), system_transcript

    def process_robot_input(self, step, interruptable, object_info,
                            move_arm, move_base, current_loc, destination_loc):
        # todo process the message from the robot to create speech output
        self.robot_step = step
        print("step", step)
        print("interruptable", interruptable)
        print("object_info", object_info)
        print("move_arm", move_arm)
        print("move_base", move_base)
        print("current_loc", current_loc)
        print("destination_loc", destination_loc)

        self.robot_interruptable = interruptable
        string_for_synthesizer = f"I am {step}"
        
        dict_object = object_info
        move_arm = move_arm
        move_base = move_base
        current_location = current_loc
        destination_location = destination_loc
        objects_in_use = self.objects_in_use
       
        print("user data", self.user_data)
        utterance_user = self.user_data["transcript"]
        age = self.user_data["age"]
        confidence_of_age = self.user_data["confidence"]
       

        string_for_synthesizer, response_to_robot = self.agent.information_process(
                                                        utterance_user, age, confidence_of_age, self.robot_step, 
                                                        self.robot_interruptable, dict_object, move_arm, move_base, 
                                                        current_location, destination_location, objects_in_use)
        
        return string_for_synthesizer

    def get_objects_in_use(self):
        # looks at state of rosparam \objects_in_use and return list of objects
        param_name = rospy.search_param('object_in_use')
        object_str = rospy.get_param(param_name)
        self.objects_in_use = object_str.split(',')

    def get_robot_states(self):
        return self.robot_step, self.robot_interruptable