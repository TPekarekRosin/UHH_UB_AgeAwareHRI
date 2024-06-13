from playsound import playsound
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain.output_parsers import RegexParser
from langchain.chat_models import ChatOpenAI
import re
from dialogue_system.prompts import prompt_1
# from prompts import prompt_1

#      ___       _______  _______ .__   __. .___________.
#     /   \     /  _____||   ____||  \ |  | |           |
#    /  ^  \   |  |  __  |  |__   |   \|  | `---|  |----`
#   /  /_\  \  |  | |_ | |   __|  |  . `  |     |  |     
#  /  _____  \ |  |__| | |  |____ |  |\   |     |  |     
# /__/     \__\ \______| |_______||__| \__|     |__|    

class SocialBrain:
    
    # @classmethod
    # def get_docs(cls, env):
    #     return env.unwrapped.__doc__
    
    # def __init__(self):
    def __init__(self, model, prompt):
        self.model = model
        self.prompt = prompt
        self.action_parser = RegexParser( 
            regex=r"^system_transcript:(,*)command: (,*)add_type: (,*)add_color: (,*)add_name: (,*)add_location: (,*)add_size: (,*)del_type: (,*)del_color: (,*)del_name: (,*)del_location: (,*)del_size",
            output_keys=["system_transcript","command","add_type","add_color","add_name","add_location","add_size","del_type","del_color","del_name","del_location","del_size"],
            default_output_key="system_transcript",
        )
        self.message_history = []
        # self.ret = 0
        
    def information_process(self, utterance_user, age, confidence_of_age, step, interruptible, dict_object, move_arm, move_base, current_location, destination_location, objects_in_use):
        
        human_message = f"user_utterance: {utterance_user} age: {age} confidence_of_age: {confidence_of_age} step: {step} interruptible: {interruptible} dict_object :{dict_object} move_arm:{move_arm} move_base:{move_base} current_location:{current_location} destination_location:{destination_location} objects_in_use:{objects_in_use}"
        print("-----------------------------before model--------------------------")
        print("human_message", human_message)
        self.message_history.append(HumanMessage(content=human_message)
        )
        # Use LLM to process the user's utterance or robot status, and then generate a transcript for the user or command for robot
        act_message = self.model(self.message_history)
        print("-----------------------------after model--------------------------")
        print("act_message:", act_message)
        print("act_message content:", act_message.content)
        # parsed = self.action_parser.parse(act_message.content)
        # Define a regular expression pattern to match key-value pairs
        pattern = re.compile(r'(\w+):\s*"([^"]*)"')
        # Find all matches in the input string
        matches = pattern.findall(act_message.content)
        # Create a dictionary from the matches
        parsed = dict(matches)
        system_transcript = str
        response_to_robot = dict()
        # print("parsed:", parsed)
        if len(parsed) == 0:
            print("the output is wrong")
            system_transcript = "the output is wrong"
        else:
            system_transcript = parsed["system_transcript"]
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
        print("-----------------------------last--------------------------")
        print("system_transcript: ", system_transcript)
        print("response_to_robot: ", response_to_robot)
        return system_transcript, response_to_robot
    def reset(self):
        self.message_history = [
            SystemMessage(content=self.prompt),
        ]
        return
    
    
if __name__ == '__main__':

    with open("openai_api_key.txt") as fapi:
        api_key = fapi.read()
    # env = SocialEnv()
    # print("env:", env)
    # model_version = "gpt-3.5-turbo-instruct"
    model_version = "gpt-3.5-turbo-1106"
    prompt = prompt_1
    chat = ChatOpenAI(temperature=0.1, verbose=True, model_name=model_version, max_tokens=256, openai_api_key=api_key)
    agent = SocialBrain(model=chat, prompt=prompt)
    agent.reset()
   
    while True:
        
        user_utterance = input("user_utterance: ")
        if user_utterance == "exit" or user_utterance == "stop" or user_utterance == "break":
            break
        # elder or young
        age = input("age: ")
        confidence_of_age = input("confidence_of_age: ")
        # already_done, set_parameters, transporting_search, transporting_fetch, and transporting_deliver.
        step = input("step: ")
        # False or True
        interruptible= input("interruptible: ")
        # "{type: null, color: null, name: null, location: null, size: null}"
        # {type: '', color: '', name: '', location: '', size: ''}
        dict_object = input("dict_object: ")
        # False or True
        move_arm = input("move_arm: ")
        # False or True
        move_base = input("move_base: ")
        # cupboard, fridge, countertop, dishwasher, drawer, table
        current_location = input("current_location: ")
        destination_location = input("destination_location: ")
        objects_in_use = input("objects_in_use: ")
        
        system_transcript, response_to_robot = agent.information_process(user_utterance, age, confidence_of_age, step, interruptible, dict_object, move_arm, move_base, current_location, destination_location, objects_in_use)
         
        print("start next turn")
    
    

        
    
    