import itertools
from openai import OpenAI
import json
from prompts import prompt_data_generation

with open("openai_api_key.txt") as fapi:
        api_key = fapi.read()
client = OpenAI(api_key=api_key)

'''
Can you bring me a cup?
Would you please put the big container of milk on the table?
I need a spoon to eat my cereal. Could you fetch me one?
This is the wrong container of milk, please bring me the blue one.
Could you please bring me the cereal in the green box?
Please bring me a bowl and a spoon.
Can you get me another cup?
Would you please bring me a spoon for my tea?
I just need a bowl, no cereal today.
Can you set a cup and spoon on the table for lunch?
'''

# data ={
#         'sim_scene1': {
#             "bowl": {
#                 "color": "white",
#                 "size": "medium",
#                 "shape": "round",
#                 "container": "true",
#                 "state": "dirty",
#                 "destination": "dishwasher",
#                 "grasping_type": "edge grasp",
#                 "placing_type": "place"
#             },
#             "spoon": {
#                 "color": "silver",
#                 "size": "small",
#                 "shape": "oval",
#                 "container": "false",
#                 "state": "dirty",
#                 "destination": "dishwasher",
#                 "grasping_type": "top grasp",
#                 "placing_type": "place"
#             },
#         }
# }

# json_string = json.dumps(data, indent=4)  

# with open('/home/sun/Projects_HRD/UHH_UB_AgeAwareHRI/src/speech_processing/src/dialogue_system/dialog_data.json', 'w') as outfile:
#     outfile.write(json_string)

def instruction_generate(model_version, prompt_data_generation, object_name, object_color, object_size, object_location):
    prompts = [{"role": "system", "content": prompt_data_generation},]
    object = " object name: " + object_name + ", object color: " + object_color + ", object size: " + object_size + ", object_location: " + object_location
    messages = prompts
    object_prompts = [{"role": "user", "content": object}]
    messages = prompts + object_prompts
    response = client.chat.completions.create(
                                        model=model_version,
                                        # response_format={ "type": "json_object" },
                                        messages=messages,
                                        temperature=0,
                                    )
    instructions = response.choices[0].message.content
    instructions_json = json.loads(instructions[8:-4])
    print(instructions_json)
    return instructions_json
def read_current_data(path):
        try:
            with open(path, 'r') as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
def write_data(path, data):
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)
   

if __name__ == '__main__':
    objects = ["milk", "bowl", "cereal", "spoon", "cup"]
    colors = ["green", "blue", "red", "white", "null"]
    size = ["big", "normal", "small", "null"]
    locations = ["cupboard", "countertop", "dishwasher", "null"]
    
    combinations = list(itertools.product(objects, colors, size, locations))
    print(F"The length of combinations is {len(combinations)}.")
    dataset_path = ""

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
        instructions_json = instruction_generate(model_version, prompt_data_generation, object_name, object_color, object_size, object_location)
        formal_instruction = instructions_json['formal']
        informal_instruction = instructions_json['informal']
        new_data = {
            "input":{
                "user_utterance": formal_instruction,
                "age": "young",
                "confidence_of_age": 75,
                "step": "transporting_deliver",
                "interruptible": True,
                "dict_object": {"type": "cup", "color": "red", "name": "", "location": "", "size": ""},
                "move_arm": False,
                "move_base": False,
                "current_location": "kitchen",
                "destination": "dining area",
                "objects_in_use": []
                },
            "annotation":{
                "system_transcript": "Sure, I will move from the kitchen to the dining area.",
                "command": "change_location",
                "add_type": "", 
                "add_color": "", 
                "add_name": "", 
                "add_location": "", 
                "add_size": "",
                "del_type": "", 
                "del_color": "", 
                "del_name": "", 
                "del_location": "", 
                "del_size": "",
            }
        }
        
      

        current_data.append(new_data)
        write_data(dataset_path, current_data)
        
    


