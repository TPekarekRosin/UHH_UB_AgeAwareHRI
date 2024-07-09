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
def write_data(path, data):
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)
        
    
def calculation(datasets):
    index = 0
    sum_command = 0 
    sum_add_type = 0
    sum_add_color = 0
    sum_add_name = 0
    sum_add_size = 0
    sum_add_location = 0
    sum_del_type = 0
    sum_del_color = 0
    sum_del_name = 0
    sum_del_size = 0
    sum_del_location = 0
    
    for dataset in datasets:
        index +=1
        with open("openai_api_key.txt") as fapi:
            api_key = fapi.read()
        # model_version = "gpt-3.5-turbo-1106"
        model_version = "gpt-3.5-turbo-0125"
        prompt = prompt_1
        chat = ChatOpenAI(temperature=0.1, verbose=True, model_name=model_version, max_tokens=256, openai_api_key=api_key)
        agent = SocialBrain(model=chat, prompt=prompt)
        agent.reset()
        print("dataset", dataset)
        user_utterance = dataset["input"]["user_utterance"]
        age = dataset["input"]["age"]
        confidence_of_age = dataset["input"]["confidence_of_age"]
        step = dataset["input"]["step"] 
        interruptible= dataset["input"]["interruptible"] 
        dict_object = dataset["input"]["dict_object"]
        move_arm = dataset["input"]["move_arm"]
        move_base = dataset["input"]["move_base"]
        current_location = dataset["input"]["current_location"]
        destination = dataset["input"]["destination"]
        objects_in_use = dataset["input"]["objects_in_use"]
        
        system_transcript, response_to_robot = agent.information_process(user_utterance, age, confidence_of_age, step, interruptible, dict_object, move_arm, move_base, current_location, destination, objects_in_use)
        
        if response_to_robot is None:
            print("No valid JSON data could be parsed.")
        else:  
            print(f"response_to_robot is {response_to_robot}") 
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
            ann_add_color = dataset["annotation"]["add_color"]
            ann_add_name = dataset["annotation"]["add_name"]
            ann_add_size = dataset["annotation"]["add_size"]
            ann_add_location = dataset["annotation"]["add_location"]
            ann_del_type = dataset["annotation"]["del_type"]
            ann_del_color = dataset["annotation"]["del_color"]
            ann_del_name = dataset["annotation"]["del_name"]
            ann_del_size = dataset["annotation"]["del_size"]
            ann_del_location = dataset["annotation"]["del_location"]
            
            if pred_command == ann_command:
                sum_command +=1
            if pred_add_type == ann_add_type:    
                sum_add_type +=1
            if pred_add_color == ann_add_color:    
                sum_add_color +=1
            if pred_add_name == ann_add_name:    
                sum_add_name +=1
            if pred_add_size == ann_add_size:    
                sum_add_size +=1
            if pred_add_location == ann_add_location:    
                sum_add_location +=1
            if pred_del_type == ann_del_type:    
                sum_del_type +=1
            if pred_del_color == ann_del_color:    
                sum_del_color +=1
            if pred_del_name == ann_del_name:    
                sum_del_name +=1
            if pred_del_size == ann_del_size:    
                sum_del_size +=1
            if pred_del_location == ann_del_location:   
                sum_del_location +=1
                
            print(f"sum_command is {sum_command}")
            print(f"sum_add_type is {sum_add_type}")
            print(f"sum_add_color is {sum_add_color}")
            print(f"sum_add_name is {sum_add_name}")
            print(f"sum_add_size is {sum_add_size}")
            print(f"sum_add_location is {sum_add_location}")
            print(f"sum_del_type is {sum_del_type}")
            print(f"sum_del_color is {sum_del_color}")
            print(f"sum_del_name is {sum_del_name}")
            print(f"sum_del_size is {sum_del_size}")
            print(f"sum_del_location is {sum_del_location}")
            print(f"index is {index}")

    
if __name__ == '__main__':

    dataset_path1 = "/home/sun/Projects_Learning/UHH_UB_AgeAwareHRI/src/speech_processing/src/dialogue_system/one_turn_set_bringme.json"
    dataset_path2 = "/home/sun/Projects_Learning/UHH_UB_AgeAwareHRI/src/speech_processing/src/dialogue_system/one_turn_set_breakfast.json"
    dataset_path3 = "/home/sun/Projects_Learning/UHH_UB_AgeAwareHRI/src/speech_processing/src/dialogue_system/one_turn_set_replace.json"
    datasets_path = "/home/sun/Projects_Learning/UHH_UB_AgeAwareHRI/src/speech_processing/src/dialogue_system/one_turn_set.json"
    
    # dataset1 = read_current_data(dataset_path1)
    # dataset2 = read_current_data(dataset_path2)
    # dataset3 = read_current_data(dataset_path3)
    # datasets = dataset1 + dataset2 + dataset3
    # random.shuffle(datasets)
    # write_data(datasets_path, datasets)
    datasets = read_current_data(datasets_path)
    calculation(datasets)

    print("done")