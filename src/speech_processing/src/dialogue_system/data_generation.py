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
        

    # template1 = "Could you please bring me a cup?"
    # template2 = "Hey, can you get me a cup?"
    
    # template3 = "Would you mind bringing me a cup?"
    # template4 = "Bring me a cup now."
    
    # template5 = "Would you grab me a cup?"
    # template6 = "Fetch me a cup, please."
    
    # template7 = "Could you provide me with a cup?"
    # template8 = "Can you hand me a cup?"
    
    # template9 = "May I request a cup from you?"
    # template10 = "I need a cup immediately."

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
        
        if object_color != "null" and object_size != "null" and object_location != "null":
            instruction1 = f"Could you please bring me a {object_size} {object_color} {object_name} from {object_location}?"
            instruction2 = f"Hey, can you get me a {object_size} {object_color} {object_name} from {object_location}?"
            
        elif object_color == "null" and object_size == "null" and object_location != "null": 
            instruction1 = f"Would you mind bringing me a {object_name} from {object_location}?"
            instruction2 = f"Bring me a {object_name} from {object_location} now."
            
        elif object_color == "null" and object_size != "null" and object_location == "null": 
            instruction1 = f"Would you grab me a {object_size} {object_name}?"
            instruction2 = f"Fetch me a {object_size} {object_name}, please."
            
        elif object_color != "null" and object_size == "null" and object_location == "null": 
            instruction1 = f"Could you provide me with a {object_color} {object_name}?"
            instruction2 = f"Can you hand me a {object_color} {object_name}?"
            
        elif object_color == "null" and object_size == "null" and object_location == "null":
            instruction1 = f"May I request a {object_name} from you?"
            instruction2 = f"I need a {object_name} immediately."
            
        system_transcript = "ok, wait a moment"
        new_data1 = {
                        "input":{
                            "user_utterance": instruction1,
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
                            "user_utterance": instruction2,
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

def replace_object(objects, colors, size):
    
    # Generate all possible combinations
    combinations = list(itertools.product(objects, colors, size))
    print(F"The length of combinations is {len(combinations)}.")
    # Generate all possible pairs of combinations
    pair_combinations = list(itertools.combinations(combinations, 2))
    print(f"the length of pairs is {len(pair_combinations)}")
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
        
         
        instruction1 = f"I would prefer a {object_size1} {object_color1} {object_name1} over a {object_size2} {object_color2} {object_name2}"
        instruction2 = f"Could I have a {object_size1} {object_color1} {object_name1} instead of a {object_size2} {object_color2} {object_name2}?"
        instruction3 = f"I would like a {object_size1} {object_color1} {object_name1} rather than a {object_size2} {object_color2} {object_name2}."
        # 
        # I would like to have a cup in place of a bowl.
        # I would choose a cup over a bowl.
        # I want to switch the bowl for a cup.
        # Please give me a cup instead of a bowl.
        # Can you substitute a cup for the bowl?
        # I need a cup, not a bowl.
        # I would prefer to have a cup instead of a bowl.
        # I want a cup rather than a bowl.
        # I want a cup in place of a bowl.
        # I want a cup over a bowl.
        # I want a cup in lieu of a bowl.
        # I want a cup in preference to a bowl.


        
       
        
        # system_transcript = 'ok, wait moment.'
        
        # new_data1 = {
        #                 "input":{
        #                     "user_utterance": instruction1,
        #                     "age": 'young',
        #                     "confidence_of_age": 90,
        #                     "step": '',
        #                     "interruptible": True,
        #                     "dict_object": {"type": '', "color": '', "name": '', "location": '', "size": ''},
        #                     "move_arm": False,
        #                     "move_base": False,
        #                     "current_location": '',
        #                     "destination": '',
        #                     "objects_in_use": []
        #                     },
        #                 "annotation":{
        #                     "system_transcript": system_transcript,
        #                     "command": 'replace_object',
        #                     "add_type": object_name1, 
        #                     "add_color": object_color1, 
        #                     "add_name": '', 
        #                     "add_location": '',
        #                     "add_size": object_size1,
        #                     "del_type": '', 
        #                     "del_color": '', 
        #                     "del_name": '', 
        #                     "del_location": '', 
        #                     "del_size": '',
        #                     }
        #             }
        # new_data2 = {
        #                 "input":{
        #                     "user_utterance": instruction2,
        #                     "age": 'young',
        #                     "confidence_of_age": 90,
        #                     "step": '',
        #                     "interruptible": True,
        #                     "dict_object": {"type": '', "color": '', "name": '', "location": '', "size": ''},
        #                     "move_arm": False,
        #                     "move_base": False,
        #                     "current_location": '',
        #                     "destination": '',
        #                     "objects_in_use": []
        #                     },
        #                 "annotation":{
        #                     "system_transcript": system_transcript,
        #                     "command": 'replace_object',
        #                     "add_type": object_name1, 
        #                     "add_color": object_color1, 
        #                     "add_name": '', 
        #                     "add_location": '', 
        #                     "add_size": object_size1,
        #                     "del_type": '', 
        #                     "del_color": '', 
        #                     "del_name": '', 
        #                     "del_location": '', 
        #                     "del_size": '',
        #                     }
        #             }
        # current_data.append(new_data1)
        # current_data.append(new_data2)
        # write_data(dataset_path, current_data)
        
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
    
    # Polite Request: "Could you please prepare breakfast for me?"
    # Casual Ask: "Can you make breakfast for me?"
    # Indirect Request: "I'm feeling hungry, could you help me with breakfast?"
    # Formal: "Would it be possible for you to prepare breakfast for me?"
    # Friendly: "Hey, would you mind making breakfast for me?"
    # Urgent: "I really need breakfast soon; could you make it for me?"
    # Indirect Hint: "It would be great to have some breakfast prepared."
    # Respectful: "May I ask you to prepare breakfast for me?"
    # Grateful Tone: "I would really appreciate it if you could make breakfast for me."
    # Straightforward: "Can you prepare breakfast for me, please?"
    # Playful: "How about you whip up some breakfast for me?"
    # Request with Reason: "I'm running late; could you make breakfast for me?"
    # Affectionate: "Could you make breakfast for me, sweetheart?"
    # Commanding (if appropriate): "Please prepare breakfast for me."
    # Colloquial: "Can you fix me some breakfast?"
    # Collaborative: "Shall we make breakfast together? Can you start it for me?"
    # Professional: "Would you be able to arrange breakfast for me?"
    # Expressing Need: "I need some breakfast; can you help me out?"
    # Hinting: "I would love some breakfast if you have the time."
    # Request with Praise: "Your breakfasts are always the best; could you make one for me?"
    
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
    
    # bring_me_command(objects, colors1, size1, locations1)
    
    replace_object(objects, colors2, size2)
    
    # setting_breakfast()
    
        


