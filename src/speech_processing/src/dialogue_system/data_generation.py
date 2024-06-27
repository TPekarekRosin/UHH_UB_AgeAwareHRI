from openai import OpenAI
import json

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

data ={
        'sim_scene1': {
            "bowl": {
                "color": "white",
                "size": "medium",
                "shape": "round",
                "container": "true",
                "state": "dirty",
                "destination": "dishwasher",
                "grasping_type": "edge grasp",
                "placing_type": "place"
            },
            "spoon": {
                "color": "silver",
                "size": "small",
                "shape": "oval",
                "container": "false",
                "state": "dirty",
                "destination": "dishwasher",
                "grasping_type": "top grasp",
                "placing_type": "place"
            },
        }
}

json_string = json.dumps(data, indent=4)  

with open('/home/sun/Projects_HRD/UHH_UB_AgeAwareHRI/src/speech_processing/src/dialogue_system/dialog_data.json', 'w') as outfile:
    outfile.write(json_string)

prompt = [
    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
    {"role": "user", "content": "Who won the world series in 2020?"}]

# Example OpenAI Python library request
# MODEL = "gpt-3.5-turbo"
# MODEL ="gpt-4-vision-preview"
MODEL ="gpt-4o-2024-05-13"
response = client.chat.completions.create(
    model=MODEL,
    response_format={ "type": "json_object" },
    messages=prompt,
    temperature=0,
)
output = response.choices[0].message.content
output_json = json.loads(output)
print(output_json)


