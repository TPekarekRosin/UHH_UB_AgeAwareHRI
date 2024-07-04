import itertools
import json
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
    dataset_path = "/home/sun/Projects_HRD/UHH_UB_AgeAwareHRI/src/speech_processing/src/dialogue_system/one_turn_set_bringme.json"
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
                            "step": 'set_parameters',
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
                            "step": 'set_parameters',
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
    dataset_path = "/home/sun/Projects_HRD/UHH_UB_AgeAwareHRI/src/speech_processing/src/dialogue_system/one_turn_set_replace.json"
    
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
        instructions = [f"I would prefer a {object_size1} {object_color1} {object_name1} over a {object_size2} {object_color2} {object_name2}",
        f"Could I have a {object_size1} {object_color1} {object_name1} instead of a {object_size2} {object_color2} {object_name2}?",
        f"I would like a {object_size1} {object_color1} {object_name1} rather than a {object_size2} {object_color2} {object_name2}.",
        f"I would like to have a {object_size1} {object_color1} {object_name1}  in place of a {object_size2} {object_color2} {object_name2}.",
        f"I would choose a {object_size1} {object_color1} {object_name1} over a {object_size2} {object_color2} {object_name2}.",
        f"I want to switch the {object_size1} {object_color1} {object_name1} for a {object_size2} {object_color2} {object_name2}.",
        f"Please give me a {object_size1} {object_color1} {object_name1} instead of a {object_size2} {object_color2} {object_name2}.",
        f"Can you substitute a {object_size1} {object_color1} {object_name1} for the {object_size2} {object_color2} {object_name2}?",
        f"I need a {object_size1} {object_color1} {object_name1}, not a {object_size2} {object_color2} {object_name2}.",
        f"I would prefer to have a {object_size1} {object_color1} {object_name1}  instead of a {object_size2} {object_color2} {object_name2}.",
        f"I want a {object_size1} {object_color1} {object_name1} rather than a {object_size2} {object_color2} {object_name2}.",
        f"I want a {object_size1} {object_color1} {object_name1} in place of a {object_size2} {object_color2} {object_name2}.",
        f"I want a {object_size1} {object_color1} {object_name1} over a {object_size2} {object_color2} {object_name2}.",
        f"I want a {object_size1} {object_color1} {object_name1} in lieu of a {object_size2} {object_color2} {object_name2}.",
        f"I want a {object_size1} {object_color1} {object_name1} in preference to a {object_size2} {object_color2} {object_name2}."]
        system_transcript = 'ok, wait moment.'
        random_number = random.randint(0, 14)
        new_data = {
                        "input":{
                            "user_utterance": instructions[random_number],
                            "age": 'young',
                            "confidence_of_age": 90,
                            "step": 'set_parameters',
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
                            "del_type": object_name2, 
                            "del_color": object_color2, 
                            "del_name": '', 
                            "del_location": '', 
                            "del_size": object_size2,
                            }
                    }
        current_data.append(new_data)
        write_data(dataset_path, current_data)
        
def setting_breakfast():
    
    utterances = ["Could you please make breakfast?",
    "Can you whip up some breakfast?",
    "Would you be so kind as to prepare breakfast?",
    "Make breakfast, please.",
    "I am really hungry. Breakfast sounds good right now.",
    "It would be great to have breakfast ready soon.",
    "I could really use some breakfast. Do you mind making it?",
    "How about we prepare breakfast together?",
    "Could you make breakfast today?",
    "Would you mind making us breakfast?",
    "Can you make breakfast this time?",
    "I would really appreciate it if you could make breakfast.",
    "Thank you in advance for making breakfast.",
    "Could you make some breakfast?",
    "Hey, can you make breakfast?",
    "Mind making breakfast?",
    "Can you be the breakfast chef today?",
    "It is morning! How about some breakfast?",
    "Could you have breakfast ready by 8 AM?",
    "Can you handle breakfast today?",
    "You make the best breakfast! Can you make it today?",
    "Could you please prepare breakfast for me?",
    "Can you make breakfast for me?",
    "I'm feeling hungry, could you help me with breakfast?",
    "Would it be possible for you to prepare breakfast for me?",
    "Hey, would you mind making breakfast for me?",
    "I really need breakfast soon; could you make it for me?",
    "It would be great to have some breakfast prepared.",
    "May I ask you to prepare breakfast for me?",
    "I would really appreciate it if you could make breakfast for me.",
    "Can you prepare breakfast for me, please?",
    "How about you whip up some breakfast for me?",
    "I'm running late, could you make breakfast for me?",
    "Could you make breakfast for me, sweetheart?",
    "Please prepare breakfast for me.",
    "Can you fix me some breakfast?",
    "Shall we make breakfast together? Can you start it for me?",
    "Would you be able to arrange breakfast for me?",
    "I need some breakfast, can you help me out?",
    "I would love some breakfast if you have the time.",
    "Your breakfasts are always the best, could you make one for me?"]
    
    print(f"the length of utterence is {len(utterances)}")
    
    system_transcript = "I will prepare the table for your breakfast"
    dataset_path = "/home/sun/Projects_HRD/UHH_UB_AgeAwareHRI/src/speech_processing/src/dialogue_system/one_turn_set_breakfast.json"
    
    for utterance in utterances:
        current_data = read_current_data(dataset_path)
        new_data = {
                        "input":{
                            "user_utterance": utterance,
                            "age": 'young',
                            "confidence_of_age": 90,
                            "step": 'set_parameters',
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
    # replace_object(objects, colors2, size2)
    setting_breakfast()
   
    
    
    
        


