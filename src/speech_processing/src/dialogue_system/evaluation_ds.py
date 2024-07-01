# from dialogue_system.prompts import prompt_1
from prompts import prompt_1
from langchain.chat_models import ChatOpenAI

from social_brain import SocialBrain














    
if __name__ == '__main__':

    with open("openai_api_key.txt") as fapi:
        api_key = fapi.read()
    # env = SocialEnv()P
    # print("env:", env)
    # model_version = "gpt-3.5-turbo-instruct"
    model_version = "gpt-3.5-turbo-1106"
    prompt = prompt_1
    chat = ChatOpenAI(temperature=0.1, verbose=True, model_name=model_version, max_tokens=256, openai_api_key=api_key)
    agent = SocialBrain(model=chat, prompt=prompt)
    agent.reset()
   
    # while True:
        
    #     user_utterance = input("user_utterance: ")
    #     if user_utterance == "exit" or user_utterance == "stop" or user_utterance == "break":
    #         break
    #     # elder or young
    #     age = input("age: ")
    #     confidence_of_age = input("confidence_of_age: ")
    #     # already_done, set_parameters, transporting_search, transporting_fetch, and transporting_deliver.
    #     step = input("step: ")
    #     # False or True
    #     interruptible= input("interruptible: ")
    #     # "{type: null, color: null, name: null, location: null, size: null}"
    #     # {type: '', color: '', name: '', location: '', size: ''}
    #     dict_object = input("dict_object: ")
    #     # False or True
    #     move_arm = input("move_arm: ")
    #     # False or True
    #     move_base = input("move_base: ")
    #     # cupboard, fridge, countertop, dishwasher, drawer, table
    #     current_location = input("current_location: ")
    #     destination_location = input("destination_location: ")
    #     objects_in_use = input("objects_in_use: ")
        
    #     system_transcript, response_to_robot = agent.information_process(user_utterance, age, confidence_of_age, step, interruptible, dict_object, move_arm, move_base, current_location, destination_location, objects_in_use)
         
    #     print("start next turn")