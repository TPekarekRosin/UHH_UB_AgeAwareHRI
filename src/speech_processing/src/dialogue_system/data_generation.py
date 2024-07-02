import itertools
from openai import OpenAI
import json
from prompts import prompt_bring_me_data_generation, prompt_replace_object_generation

with open("openai_api_key.txt") as fapi:
        api_key = fapi.read()
client = OpenAI(api_key=api_key)

def read_current_data(path):
        try:
            with open(path, 'r') as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
def write_data(path, data):
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)

def bring_me_instruction_generation(model_version, prompt_bring_me_data_generation, object_name, object_color, object_size, object_location):
    prompts = [{"role": "system", "content": prompt_bring_me_data_generation},
                {"role": "user", "content": "object name: cup, object color: blue, object size: small, object location: cupboard"},
               {"role": "assistant", "content": "{'formal': 'I would like a small red cup from cupboard.', 'informal': ' Can I get a small red cup from cupboard.'"},]
    object = " object name: " + object_name + ", object color: " + object_color + ", object size: " + object_size + ", object location: " + object_location
    object_prompts = [{"role": "user", "content": object}]
    messages = prompts + object_prompts
    response = client.chat.completions.create(
                                        model=model_version,
                                        messages=messages,
                                        temperature=0,
                                    )
    instructions = response.choices[0].message.content
    
    instructions_json = json.loads(instructions[8:-4])
    print(instructions_json)
    return instructions_json

def bring_me_command(objects, colors, size, locations):
    combinations = list(itertools.product(objects, colors, size, locations))
    print(F"The length of combinations is {len(combinations)}.")
    dataset_path = "/home/sun/Projects_HRD/UHH_UB_AgeAwareHRI/src/speech_processing/src/dialogue_system/one_turn_set.json"
    for combo in combinations:
        object_name = combo[0]
        object_color = combo[1]
        object_size = combo[2]
        object_location = combo[3]
        current_data = read_current_data(dataset_path)
    
        # Example OpenAI Python library request
        # model_version = "gpt-3.5-turbo"
        # model_version ="gpt-4-vision-preview"
        model_version ="gpt-4o-2024-05-13" 
        instructions_json = bring_me_instruction_generation(model_version, prompt_bring_me_data_generation, object_name, object_color, object_size, object_location)
        formal_instruction = instructions_json['formal']
        informal_instruction = instructions_json['informal']
        if object_color != "null" and object_size != "null" and object_location != "null":
            system_transcript = 'I will bring you a ' + object_size + " " + object_color + " " + object_name + " from " + object_location
        elif object_color == "null" and object_size == "null" and object_location != "null": 
            system_transcript = 'I will bring you a ' + object_name + " from " + object_location
        elif object_color == "null" and object_size != "null" and object_location == "null": 
            system_transcript = 'I will bring you a ' + object_size + " " + object_name
        elif object_color != "null" and object_size == "null" and object_location == "null": 
                system_transcript = 'I will bring you a ' + object_color + " " + object_name
        elif object_color == "null" and object_size == "null" and object_location == "null":
            system_transcript = 'I will bring you a '+ " " + object_name
        new_data1 = {
                        "input":{
                            "user_utterance": formal_instruction,
                            "age": 'young',
                            "confidence_of_age": 90,
                            "step": '',
                            "interruptible": True,
                            "dict_object": {"type": '', "color": '', "name": '', "location": '', "size": ''},
                            "move_arm": False,
                            "move_base": False,
                            "current_location": '',
                            "destination": '',
                            "objects_in_use": []
                            },
                        "annotation":{
                            "system_transcript": system_transcript,
                            "command": 'bring_me',
                            "add_type": object_name, 
                            "add_color": object_color, 
                            "add_name": '', 
                            "add_location": object_location, 
                            "add_size": '',
                            "del_type": '', 
                            "del_color": '', 
                            "del_name": '', 
                            "del_location": '', 
                            "del_size": '',
                            }
                    }
        new_data2 = {
                        "input":{
                            "user_utterance": informal_instruction,
                            "age": 'young',
                            "confidence_of_age": 90,
                            "step": '',
                            "interruptible": True,
                            "dict_object": {"type": '', "color": '', "name": '', "location": '', "size": ''},
                            "move_arm": False,
                            "move_base": False,
                            "current_location": '',
                            "destination": '',
                            "objects_in_use": []
                            },
                        "annotation":{
                            "system_transcript": system_transcript,
                            "command": 'bring_me',
                            "add_type": object_name, 
                            "add_color": object_color, 
                            "add_name": '', 
                            "add_location": object_location, 
                            "add_size": '',
                            "del_type": '', 
                            "del_color": '', 
                            "del_name": '', 
                            "del_location": '', 
                            "del_size": '',
                            }
                    }
        current_data.append(new_data1)
        current_data.append(new_data2)
        write_data(dataset_path, current_data)

def replace_object_instruction_generation(model_version, prompt_replace_object_generation, object_name1, object_color1, object_size1, object_name2, object_color2, object_size2):
    prompts = [{"role": "system", "content": prompt_replace_object_generation},
               {"role": "user", "content": "target object name: cup, target object color: blue, target object size: small, delete object name: bowl, delete object color: red, delete object size: small"},
               {"role": "assistant", "content": "{'formal': 'I would prefer a small blue cup rather than a small red bowl', 'informal': 'Can I get a small blue cup instead of a small red bowl?'"},]
    objects = "target object name: " + object_name1 + ", target object color: " + object_color1 + ", target object size: " + object_size1 + ", delete object name: " + object_name2 + ", delete object color: " + object_color2 + ", delete object size: " + object_size2
    object_prompts = [{"role": "user", "content": objects}]
    messages = prompts + object_prompts
    # print(f"messages is {messages}")
    response = client.chat.completions.create(
                                            model=model_version,
                                            messages=messages,
                                            temperature=0,
                                        )
    instructions = response.choices[0].message.content
    
    instructions_json = json.loads(instructions[8:-4])
    print(instructions_json)
    return instructions_json

def replace_object(objects, colors, size):
    
    # Generate all possible combinations
    combinations = list(itertools.product(objects, colors, size))
    print(F"The length of combinations is {len(combinations)}.")
    # Generate all possible pairs of combinations
    pair_combinations = list(itertools.combinations(combinations, 2))
    print(f"the length of pairs is {pair_combinations}")
    dataset_path = "/home/sun/Projects_HRD/UHH_UB_AgeAwareHRI/src/speech_processing/src/dialogue_system/one_turn_set.json"
    
    # Print the pairs of combinations
    for pair in pair_combinations:
        # target object
        object_name1 = pair[0][0]
        object_color1 = pair[0][1]
        object_size1 = pair[0][2]
        # delete object
        object_name2 = pair[1][0]
        object_color2 = pair[1][1]
        object_size2 = pair[1][2]
        
        current_data = read_current_data(dataset_path)
        
        # Example OpenAI Python library request
        # model_version = "gpt-3.5-turbo"
        # model_version ="gpt-4-vision-preview"
        model_version = "gpt-4o-2024-05-13" 
        instructions_json = replace_object_instruction_generation(model_version, prompt_replace_object_generation, object_name1, object_color1, object_size1, object_name2, object_color2, object_size2)
        formal_instruction = instructions_json['formal']
        informal_instruction = instructions_json['informal']
        system_transcript = 'ok, wait moment, I will bring you a ' + object_size1 + " " + object_color1 + " " + object_name1
        
        new_data1 = {
                        "input":{
                            "user_utterance": formal_instruction,
                            "age": 'young',
                            "confidence_of_age": 90,
                            "step": '',
                            "interruptible": True,
                            "dict_object": {"type": '', "color": '', "name": '', "location": '', "size": ''},
                            "move_arm": False,
                            "move_base": False,
                            "current_location": '',
                            "destination": '',
                            "objects_in_use": []
                            },
                        "annotation":{
                            "system_transcript": system_transcript,
                            "command": 'replace_object',
                            "add_type": object_name1, 
                            "add_color": object_color1, 
                            "add_name": '', 
                            "add_location": '',
                            "add_size": object_size1,
                            "del_type": '', 
                            "del_color": '', 
                            "del_name": '', 
                            "del_location": '', 
                            "del_size": '',
                            }
                    }
        new_data2 = {
                        "input":{
                            "user_utterance": informal_instruction,
                            "age": 'young',
                            "confidence_of_age": 90,
                            "step": '',
                            "interruptible": True,
                            "dict_object": {"type": '', "color": '', "name": '', "location": '', "size": ''},
                            "move_arm": False,
                            "move_base": False,
                            "current_location": '',
                            "destination": '',
                            "objects_in_use": []
                            },
                        "annotation":{
                            "system_transcript": system_transcript,
                            "command": 'replace_object',
                            "add_type": object_name1, 
                            "add_color": object_color1, 
                            "add_name": '', 
                            "add_location": '', 
                            "add_size": object_size1,
                            "del_type": '', 
                            "del_color": '', 
                            "del_name": '', 
                            "del_location": '', 
                            "del_size": '',
                            }
                    }
        current_data.append(new_data1)
        current_data.append(new_data2)
        write_data(dataset_path, current_data)
        
def setting_breakfast():
    utterance1 = "Could you please make breakfast?"
    utterance2 = "Can you whip up some breakfast?"
    utterance3 = "Would you be so kind as to prepare breakfast?"
    utterance4 = "Make breakfast, please."
    utterance5 = "I am really hungry. Breakfast sounds good right now."
    utterance6 = "It would be great to have breakfast ready soon."
    utterance7 = "I could really use some breakfast. Do you mind making it?"
    utterance8 = "How about we prepare breakfast together?"
    utterance9 = "Could you make breakfast today?"
    utterance10 = "Would you mind making us breakfast?"
    utterance11 = "Can you make breakfast this time?"
    utterance12 = "I would really appreciate it if you could make breakfast."
    utterance13 = "Thank you in advance for making breakfast."
    utterance14 = "Could you make some breakfast?"
    utterance15 = "Hey, can you make breakfast?"
    utterance16 = "Mind making breakfast?"
    utterance17 = "Can you be the breakfast chef today?"
    utterance18 = "It is morning! How about some breakfast?"
    utterance19 = "Could you have breakfast ready by 8 AM?"
    utterance20 = "Can you handle breakfast today?"
    utterance21 = "You make the best breakfast! Can you make it today?"
    system_transcript = "I will prepare the table for your breakfast"
    
    dataset_path = "/home/sun/Projects_HRD/UHH_UB_AgeAwareHRI/src/speech_processing/src/dialogue_system/one_turn_set.json"
    current_data = read_current_data(dataset_path)
    new_data = {
                    "input":{
                        "user_utterance": utterance1,
                        "age": 'young',
                        "confidence_of_age": 90,
                        "step": '',
                        "interruptible": True,
                        "dict_object": {"type": '', "color": '', "name": '', "location": '', "size": ''},
                        "move_arm": False,
                        "move_base": False,
                        "current_location": '',
                        "destination": '',
                        "objects_in_use": []
                        },
                    "annotation":{
                        "system_transcript": system_transcript,
                        "command": 'setting_breakfast',
                        "add_type": '', 
                        "add_color": '', 
                        "add_name": '', 
                        "add_location": '',
                        "add_size": '',
                        "del_type": '', 
                        "del_color": '', 
                        "del_name": '', 
                        "del_location": '', 
                        "del_size": '',
                        }
                }
    current_data.append(new_data)
    write_data(dataset_path, current_data)
    

if __name__ == '__main__':
    objects = ["milk", "bowl", "cereal", "spoon", "cup"]
    colors1 = ["green", "blue", "red", "white", "null"]
    size1 = ["big", "normal", "small", "null"]
    locations1 = ["cupboard", "countertop", "dishwasher", "null"]
    
    colors2 = ["green", "blue", "red", "white"]
    size2 = ["big", "normal", "small"]
    
    bring_me_command(objects, colors1, size1, locations1)
    
    # replace_object(objects, colors2, size2)
    
    # setting_breakfast()
    
    # # # target object
    # object_name1 = 'cup'
    # object_color1 = 'blue'
    # object_size1 = 'small'
    # object_location = 'cupboard'
    # # # delete object
    # # object_name2 = 'cup'
    # # object_color2 = 'blue'
    # # object_size2 = 'big'
    # model_version = "gpt-4o-2024-05-13" 
    # # replace_object_instruction_generation(model_version, prompt_replace_object_generation, object_name1, object_color1, object_size1, object_name2, object_color2, object_size2)
        
    # bring_me_instruction_generation(model_version, prompt_replace_object_generation, object_name1, object_color1, object_size1, object_location)
        
            
        


