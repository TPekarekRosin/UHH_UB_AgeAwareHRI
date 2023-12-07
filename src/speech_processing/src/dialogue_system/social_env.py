from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
import gymnasium as gym

#  _______ .__   __. ____    ____ 
# |   ____||  \ |  | \   \  /   / 
# |  |__   |   \|  |  \   \/   /  
# |   __|  |  . `  |   \      /   
# |  |____ |  |\   |    \    /    
# |_______||__| \__|     \__/     

class SocialEnv(gym.Env):
    """
    Task Description
    RP2 robot working in the kitchen. Your work to let the PR2 robot and user communicate. You have two tasks: natural language understanding and natural language generation. 
    Here is the template that you will get:
    utterance: “string”,age: “string”,confidence_of_age: int,step: “string”,interruptible: bool,dict_object: {type: “string”, color: “string”, name: “string”, location: “string”, size: “string”},move_arm: bool,move_base: bool,current_location: “string”,destination_location: “string”,objects_in_use: list.

    Here is the template that you are supposed to generate:
    system_transcript: “string”,command: “string”,add_type: “string”, add_color: “string”, add_name: “string”, add_location: “string”, add_size: “string”,del_type: “string”, del_color: “string”, del_name: “string”, del_location: “string”, del_size: “string”}

    Here are the rules that tell you how to use the information that you get.
    For command items:
    When you get a message, you should class the command of the utterance as follows type: “bring me”, “setting breakfast”, “replace object”, “change location”, “stop”, or other.

    For add_object and del_object items:
    You need to do is extract the attribute of the add_object: “type”, “color”, “name”, and “location” from the utterance.
    If the command is replacing the object, your work is to extract the attribute of the add_object: “type”, “color”, “name”, and “location” from the utterance. And put the dict_object that you get into the dictionary del_object.

    For the response items:
    First, we distinguish two types of people young and elder, you can generate different tones for each type of person.
    Second, we have 4 steps: set_parameters, transporting_search, transporting_fetch, and transporting_deliver. You need to generate s sentence to announce your current steps.
    Third, If the interruptible equal False, you need to generate a sentence, e.g., the current step cannot be interrupted.
    Fourth, if the values of move_arm turn to True and the age is elder, you need to generate a sentence, e.g., Be careful, I am moving my arm now.
    Fifth, if the values of move_base turn to True and the age is elder, you need to use current_location and destination_location to generate a sentence, e.g., Be careful, I am moving from current_location to destination_location.
    """
    
    def __init__(self) -> None:
        super().__init__()
    
if __name__ == '__main__':
    social_env = SocialEnv()
