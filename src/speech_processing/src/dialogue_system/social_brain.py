import json

import yaml
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain.output_parsers import RegexParser
import re
# from prompts import prompt_1
# from dialogue_system.prompts import prompt_1
from langchain.chat_models import ChatOpenAI
from datetime import datetime

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
    
    def read_current_data(self, path):
        try:
            with open(path, 'r') as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    def write_data(self, path, data):
        with open(path, 'w') as file:
            json.dump(data, file, indent=4)
    def parse_json_from_string(self, data_string):
        # Attempt to load the JSON directly
        try:
            json_data = json.loads(data_string)
            return json_data
        except json.JSONDecodeError:
            # If direct loading fails, try to strip extraneous characters and then load
            try:
                stripped_data = data_string.strip()[8:-4]  # Adjust slice based on your specific needs
                json_data = json.loads(stripped_data)
                return json_data
            except json.JSONDecodeError as e:
                raise ValueError(f"Error decoding JSON: {e}")
          
        
    def information_process(self, utterance_user, age, confidence_of_age, step, interruptible, dict_object, move_arm, move_base, current_location, destination, objects_in_use):
        print(f"age value is {age}")
        if age == 0:
            age_string = "younger"
        elif age == 1:
            age_string = "elder"   
        else:
            age_string = ""   
        dataset_path = "dialog_results.json"
        current_data = self.read_current_data(dataset_path)
        
        human_message = f"user_utterance: {utterance_user}, age: {age_string}, confidence_of_age: {confidence_of_age}, step: {step}, interruptible: {interruptible}, dict_object :{dict_object}, move_arm:{move_arm}, move_base:{move_base}, current_location:{current_location}, destination:{destination}, objects_in_use:{objects_in_use}." 
        inputs = {
            'user_utterance': utterance_user, 
            'age': age_string, 
            'confidence_of_age': confidence_of_age, 
            'step': step, 
            'interruptible': interruptible, 
            'dict_object': dict_object, 
            'move_arm': move_arm, 
            'move_base': move_base, 
            'current_location': current_location, 
            'destination': destination, 
            'objects_in_use': objects_in_use
        }
        now = datetime.now()
        # print("Current date and time:", now)
        # If you just need the current time
        current_time = now.strftime("%H:%M:%S")
        current_data.append(current_time)
        current_data.append(inputs) 
        print("-----------------------------before model--------------------------")
        print("human_message", human_message)
        self.message_history.append(HumanMessage(content=human_message))
        # print(f"self.message_history is {self.message_history}")
        act_message = self.model(self.message_history)
        self.message_history.append(act_message)
        
        print("-----------------------------after model--------------------------")
        print("act_message:", act_message)
        print("act_message content:", act_message.content)
        
        try:
            results = self.parse_json_from_string(act_message.content)
            if results is None:
                print("No valid JSON data could be parsed.")
            else:
                print("JSON data parsed successfully:", results)
        except ValueError as e:
            print(e)
            return
        
        system_transcript = str
        response_to_robot = dict()
        system_transcript = results["system_transcript"]
        response_to_robot["command"] = results["command"]
        response_to_robot["add_type"] = results["add_type"]
        response_to_robot["add_color"] = results["add_color"]
        response_to_robot["add_name"] = results["add_name"]
        response_to_robot["add_location"] = results["add_location"]
        response_to_robot["add_size"] = results["add_size"]
        response_to_robot["del_type"] = results["del_type"]
        response_to_robot["del_color"] = results["del_color"]
        response_to_robot["del_name"] = results["del_name"]
        response_to_robot["del_location"] = results["del_location"]
        response_to_robot["del_location"] = results["del_location"]
        response_to_robot["del_size"] = results["del_size"]
         
        print("-----------------------------last--------------------------")
        print("system_transcript: ", system_transcript)
        print("response_to_robot: ", response_to_robot)
        current_data.append(system_transcript)
        current_data.append(response_to_robot)
        self.write_data(dataset_path, current_data)
        return system_transcript, response_to_robot
    def reset(self):
        self.message_history = [
            SystemMessage(content=self.prompt),
        ]
        return
if __name__ == '__main__':
    
    with open("openai_api_key.txt") as fapi:
        api_key = fapi.read()
    # model_version = "gpt-3.5-turbo-1106"
    # model_version = "gpt-3.5-turbo-0125"
    model_version = "gpt-4o"
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
        # {type: '', color: '', name: '', location: '', size: ''}
        dict_object = input("dict_object: ")
        # False or True
        move_arm = input("move_arm: ")
        # False or True
        move_base = input("move_base: ")
        # cupboard, fridge, countertop, dishwasher, drawer, table
        current_location = input("current_location: ")
        destination = input("destination: ")
        objects_in_use = input("objects_in_use: ")
        
        system_transcript, response_to_robot = agent.information_process(user_utterance, age, confidence_of_age, step, interruptible, dict_object, move_arm, move_base, current_location, destination, objects_in_use)
         
        print("start next turn")
    
    

        
    
    