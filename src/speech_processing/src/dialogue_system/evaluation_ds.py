# from dialogue_system.prompts import prompt_1
from prompts import prompt_1
from langchain.chat_models import ChatOpenAI
import json
from social_brain import SocialBrain
import random


def read_current_data(path):
        try:
            with open(path, 'r') as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
if __name__ == '__main__':

    with open("openai_api_key.txt") as fapi:
        api_key = fapi.read()
    # model_version = "gpt-3.5-turbo-1106"
    model_version = "gpt-4o"
    prompt = prompt_1
    chat = ChatOpenAI(temperature=0.1, verbose=True, model_name=model_version, max_tokens=256, openai_api_key=api_key)
    agent = SocialBrain(model=chat, prompt=prompt)
    agent.reset()
    
    dataset_path1 = "/home/sun/Projects_HRD/UHH_UB_AgeAwareHRI/src/speech_processing/src/dialogue_system/one_turn_set_bringme.json"
    dataset_path2 = "/home/sun/Projects_HRD/UHH_UB_AgeAwareHRI/src/speech_processing/src/dialogue_system/one_turn_set_breakfast.json"
    dataset_path3 = "/home/sun/Projects_HRD/UHH_UB_AgeAwareHRI/src/speech_processing/src/dialogue_system/one_turn_set_replace.json"
    
    dataset1 = read_current_data(dataset_path1)
    dataset2 = read_current_data(dataset_path2)
    dataset3 = read_current_data(dataset_path3)
    datasets = dataset1 + dataset2 + dataset3
    
    random.shuffle(datasets)
   
    for dataset in datasets:
        
        user_utterance = dataset["input"]["user_utterance"]
        age = dataset["input"]["age"]
        confidence_of_age = dataset["input"]["confidence_of_age"]
        # already_done, set_parameters, transporting_search, transporting_fetch, and transporting_deliver.
        step = dataset["input"]["step"]
        # False or True
        interruptible= dataset["input"]["interruptible"]
        # "{type: null, color: null, name: null, location: null, size: null}"
        # {type: '', color: '', name: '', location: '', size: ''}
        dict_object = dataset["input"]["dict_object"]
        # False or True
        move_arm = dataset["input"]["move_arm"]
        # False or True
        move_base = dataset["input"]["move_base"]
        # cupboard, fridge, countertop, dishwasher, drawer, table
        current_location = dataset["input"]["current_location"]
        destination_location = dataset["input"]["destination"]
        objects_in_use = dataset["input"]["objects_in_use"]
        
        system_transcript, response_to_robot = agent.information_process(user_utterance, age, confidence_of_age, step, interruptible, dict_object, move_arm, move_base, current_location, destination_location, objects_in_use)
         
        pred_command = response_to_robot["command"]
        
        pred_add_type = response_to_robot["add_type"]
        pred_add_color = response_to_robot["add_color"]
        pred_add_name = response_to_robot["add_name"]
        pred_add_size = response_to_robot["add_size"]
        pred_add_location = response_to_robot["add_location"]

        pred_del_type = response_to_robot["del_type"]
        pred_del_color = response_to_robot["del_color"]
        pred_del_name = response_to_robot["del_name"]
        pred_del_size = response_to_robot["del_size"]
        pred_del_location = response_to_robot["del_location"]
        
        
        ann_command = dataset["annotation"]["command"]
        
        ann_add_type = dataset["annotation"]["add_type"]
        ann__add_color = dataset["annotation"]["add_color"]
        ann__add_name = dataset["annotation"]["add_name"]
        ann__add_size = dataset["annotation"]["add_size"]
        ann__add_location = dataset["annotation"]["add_location"]

        ann__del_type = dataset["annotation"]["del_type"]
        ann__del_color = dataset["annotation"]["del_color"]
        ann__del_name = dataset["annotation"]["del_name"]
        ann__del_size = dataset["annotation"]["del_size"]
        ann__del_location = dataset["annotation"]["del_location"]

        print("start next turn")