import rospy
# from speech_processing.msg import *

from langchain.chat_models import ChatOpenAI
from dialogue_system.social_brain import SocialBrain
import ast
from dialogue_system.prompts import prompt_1


class DialogueSystem:
    def __init__(self):
        self.robot_step = None
        self.robot_interruptable = False
        self.objects_in_use = []
        self.user_data = dict()
        # string transcript
        # int32 age
        # float32 confidence
        self.user_data["transcript"] = ""
        self.user_data["age"] = "elder"
        self.user_data["confidence"] = 90
        self.robot_data = dict()
        
        self.last_robot_step = None
        self.last_robot_interruptable = None
        self.last_robot_move_arm = None
        self.last_robot_move_base = None
        self.last_robot_current_location = None
        self.last_robot_destination = None
        
        with open("openai_api_key.txt") as fapi:
            self.api_key = fapi.read()
        self.model_version = "gpt-3.5-turbo-1106"
        # self.model_version = "gpt-4o"
        self.chat = ChatOpenAI(temperature=0.1, verbose=True, model_name=self.model_version, max_tokens=1000, openai_api_key=self.api_key)
        self.prompt = prompt_1
        self.agent = SocialBrain(model=self.chat, prompt=self.prompt)
        self.agent.reset()
        
    def process_speech_input(self, transcript, age, confidence):
        
        step = str
        interruptible = str
        dict_object = dict()
        move_arm = str
        move_base = str
        current_location = str
        destination = str
        print("robot data", self.robot_data)
        if not self.robot_data:
            system_transcript = "i have issues getting the robot status"
            return '', (), system_transcript
        else:
            step = self.robot_data["step"]
            interruptible = self.robot_data["interruptable"]
            # dict_object_test = self.robot_data["object"]
            # print("dict_object_test: ", dict_object_test)
            
            dict_object["type"] = self.robot_data["object"][0].type
            dict_object["color"] = self.robot_data["object"][0].color
            dict_object["name"] = self.robot_data["object"][0].name
            dict_object["location"] = self.robot_data["object"][0].location
            dict_object["size"] = self.robot_data["object"][0].size
            print("dict_object: ", dict_object)
            
            # self.robot_data["object"]
            move_arm = self.robot_data["move_arm"]
            move_base = self.robot_data["move_base"]
            current_location = self.robot_data["current_location"]
            destination = self.robot_data["destination"]
        print("transcript", transcript)
        # todo: define minor interruptions
        if "stop" not in transcript.lower() and confidence > 0.5:
            # todo: generate valid response for minor interruptions
            system_transcript, response_to_robot = self.agent.information_process(
                                                    transcript, age, confidence, step, interruptible, 
                                                    dict_object, move_arm, move_base, current_location, 
                                                    destination, self.objects_in_use)
            self.user_data["transcript"] = ""
            self.user_data["age"] = ""
            self.user_data["confidence"] = 0
                
            
            return 'minor', response_to_robot, system_transcript
        
        # todo: define major interruptions
        elif "stop" in transcript.lower() and confidence > 0.5:
            # todo: generate valid response for major interruptions
            system_transcript = "ok i will stop anything what i am doing now"
            return 'major', (), system_transcript
        else:
            system_transcript = "i got trouble something is wrong"
            print('Command was not recognized.')
            return '', (), system_transcript

    def process_robot_input(self, step, interruptable, object_info, move_arm, move_base, current_location, destination):
        # todo process the message from the robot to create speech output
        self.robot_step = step
        self.robot_interruptable = interruptable
        string_for_synthesizer = f"I am {step}"
        
        dict_object = {}
        dict_object["type"] = object_info[0].type
        dict_object["color"] = object_info[0].color
        dict_object["name"] = object_info[0].name
        dict_object["location"] = object_info[0].location
        dict_object["size"] = object_info[0].size
        print("dict_object: ", dict_object)
       
        print("user data", self.user_data)
        utterance_user = self.user_data["transcript"]
        age = self.user_data["age"]
        confidence_of_age = self.user_data["confidence"]
        if self.last_robot_step == self.robot_step and self.last_robot_interruptable == interruptable and self.last_robot_move_arm == move_arm and self.last_robot_move_base == move_base and self.last_robot_current_location == current_location and self.last_robot_destination == destination:
            print("the robot states have not changed.")
            return    
        else:
            system_transcript, response_to_robot = self.agent.information_process(
                                                            utterance_user, age, confidence_of_age, self.robot_step, 
                                                            self.robot_interruptable, dict_object, move_arm, move_base, 
                                                            current_location, destination, self.objects_in_use)
            self.last_robot_step = self.robot_step
            self.last_robot_interruptable = self.robot_interruptable
            self.last_robot_move_arm = move_arm
            self.last_robot_move_base = move_base
            self.last_robot_current_location = current_location
            self.last_robot_destination = destination
            
            return system_transcript

    def get_robot_states(self):
        return self.robot_step, self.robot_interruptable