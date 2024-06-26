# from openai import OpenAI
# import json

# with open("openai_api_key.txt") as fapi:
#         api_key = fapi.read()
# client = OpenAI(api_key=api_key)

# x1 = """ Translate the following English text to French: "{text}" """
# x2 = [{"role": "user", "content": 'Put objects into their appropriate receptacles. objects = ["socks", "toy car", "shirt", "Lego brick"] receptacles = ["laundry basket", "storage box"] pick_and_place("socks", "laundry basket") pick_and_place("toy car", "storage box") pick_and_place("shirt", "laundry basket") pick_and_place("Lego brick", "storage box")'}]

# x3 = [
#     {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
#     {"role": "user", "content": "Who won the world series in 2020?"}
#   ]

# # print("text", w)
# # Example OpenAI Python library request
# # MODEL = "gpt-3.5-turbo"
# # MODEL ="gpt-4-vision-preview"
# MODEL ="gpt-4o-2024-05-13"
# response = client.chat.completions.create(
#     model=MODEL,
#     # response_format={ "type": "json_object" },
#     messages=x3,
#     temperature=0,
# )

# print(response.choices[0].message.content)

from openai import OpenAI

with open("openai_api_key.txt") as fapi:
    api_key = fapi.read()
client = OpenAI(api_key=api_key)

response = client.chat.completions.create(
  model="gpt-3.5-turbo-0125",
  response_format={ "type": "json_object" },
  messages=[
    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
    {"role": "user", "content": "Who won the world series in 2020?"}
  ]
)
print(response.choices[0].message.content)
