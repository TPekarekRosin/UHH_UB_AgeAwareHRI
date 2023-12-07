import gtts
from playsound import playsound
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain.output_parsers import RegexParser

from dialogue_system.social_env import SocialEnv
from langchain.chat_models import ChatOpenAI

#      ___       _______  _______ .__   __. .___________.
#     /   \     /  _____||   ____||  \ |  | |           |
#    /  ^  \   |  |  __  |  |__   |   \|  | `---|  |----`
#   /  /_\  \  |  | |_ | |   __|  |  . `  |     |  |     
#  /  _____  \ |  |__| | |  |____ |  |\   |     |  |     
# /__/     \__\ \______| |_______||__| \__|     |__|    

class SocialBrain:
    
    @classmethod
    def get_docs(cls, env):
        return env.unwrapped.__doc__
    
    # def __init__(self):
    def __init__(self, model, env):
        self.model = model
        self.docs = self.get_docs(env)
        self.action_parser = RegexParser( 
            regex=r"^system_transcript:(.*)command: (.*)add_type: (.*)add_color: (.*)add_name: (.*)add_location: (.*)add_size: (.*)del_type: (.*)del_color: (.*)del_name: (.*)del_location: (.*)del_size",
            output_keys=["system_transcript", "command", "add_type","add_color","add_name","add_location","add_size","del_type","del_color","del_name","del_location","del_size"],
            #"system_transcript", 
            # "command", 
            # "add_type",
            # "add_color",
            # "add_name",
            # "add_location",
            # "add_size",
            # "del_type",
            # "del_color",
            # "del_name",
            # "del_location",
            # "del_size",
            default_output_key="system_transcript",
        )
        self.message_history = []
        # self.ret = 0
        
    def information_process(self, utterance_user, age, confidence_of_age, step, interruptible, dict_object, move_arm, move_base, current_location, destination_location, objects_in_use):
        
        human_message = f"utterance: {utterance_user} age: {age} confidence_of_age: {confidence_of_age} step: {step} interruptible: {interruptible} dict_object :{dict_object} move_arm:{move_arm} move_base:{move_base} current_location:{current_location} destination_location:{destination_location} objects_in_use:{objects_in_use}"
        print("human_message", human_message)
        self.message_history.append(HumanMessage(content=human_message)
        )
        # Use LLM to process the user's utterance or robot status, and then generate a transcript for the user or command for robot
        act_message = self.model(self.message_history)
        
        print("act_message:", act_message)
        print("act_message content:", act_message.content)
        parsed = self.action_parser.parse(act_message.content)
        print("parsed:", parsed)
        
        system_transcript = parsed["system_transcript"]
        response_to_robot = dict()
        response_to_robot["command"] = parsed["command"]
        response_to_robot["add_type"] = parsed["add_type"]
        response_to_robot["add_color"] = parsed["add_color"]
        response_to_robot["add_name"] = parsed["add_name"]
        response_to_robot["add_location"] = parsed["add_location"]
        response_to_robot["add_size"] = parsed["add_size"]
        response_to_robot["del_type"] = parsed["del_type"]
        response_to_robot["del_color"] = parsed["del_color"]
        response_to_robot["del_name"] = parsed["del_name"]
        response_to_robot["del_location"] = parsed["del_location"]
        response_to_robot["del_location"] = parsed["del_location"]
        response_to_robot["del_size"] = parsed["del_size"]
        
        self.message_history.append(act_message)
        print("system_transcript: ", system_transcript)
        print("response_to_robot: ", response_to_robot)
        return system_transcript, response_to_robot
    def reset(self):
        self.message_history = [
            SystemMessage(content=self.docs),
        ]
        return
    
    
if __name__ == '__main__':

    with open("openai_api_key.txt") as fapi:
        api_key = fapi.read()
    env = SocialEnv()
    chat = ChatOpenAI(temperature=0, verbose=True, max_tokens=256, openai_api_key=api_key)
    agent = SocialBrain(model=chat, env=env)
    agent.reset()
   
    utterance_user = ""
    age = "elder"
    confidence_of_age = 90
    step = "set_parameters"
    interruptible= False
    move_arm = True
    move_arm = False
    move_base = False
    current_location = ""
    destination_location = "null"
    objects_in_use = "null"
    dict_object = "{object_type: cup, color: red, name: null, location: null, size: null}"
    
    
    agent.information_process(utterance_user, age, confidence_of_age, step, interruptible, dict_object, move_arm, move_base, current_location, destination_location, objects_in_use)
    
    
    

        
    
    