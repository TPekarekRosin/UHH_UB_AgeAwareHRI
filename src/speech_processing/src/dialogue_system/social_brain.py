import gtts
from playsound import playsound
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain.output_parsers import RegexParser

from social_env import SocialEnv
from langchain.chat_models import ChatOpenAI

#      ___       _______  _______ .__   __. .___________.
#     /   \     /  _____||   ____||  \ |  | |           |
#    /  ^  \   |  |  __  |  |__   |   \|  | `---|  |----`
#   /  /_\  \  |  | |_ | |   __|  |  . `  |     |  |     
#  /  _____  \ |  |__| | |  |____ |  |\   |     |  |     
# /__/     \__\ \______| |_______||__| \__|     |__|    

class SocialBrain:
    
    @classmethod
    def get_docs(cls, env):
        return env.unwrapped.__doc__
    
    # def __init__(self):
    def __init__(self, model, env):
        self.model = model
        self.docs = self.get_docs(env)
        self.action_parser = RegexParser( 
            regex=r"^response: (.*)command: (.*)add_object: (.*)del_object: (.*)",
            output_keys=["response", "command", "add_object", "del_object"],
            # output_keys=["action", "object"],
            default_output_key="response",
        )
        
        self.message_history = []
        # self.ret = 0
        
    def information_process(self, utterance_user, age, confidence_of_age, step, interruptible, dict_object, move_arm, move_base, current_location, destination_location, objects_in_use):
        
        human_message = f" utterance: {utterance_user},\n age: {age},\n confidence_of_age: {confidence_of_age},\n step: {step},\n interruptible: {interruptible},\n dict_object :{dict_object},\n move_arm:{move_arm},\n move_base:{move_base},\n current_location:{current_location},\n destination_location:{destination_location},\n objects_in_use:{objects_in_use}\n"
        print("human_message", human_message)
        self.message_history.append(HumanMessage(content=human_message)
        )
        # Use LLM to process the user's utterance or robot status, and then generate a transcript for the user or command for robot
        act_message = self.model(self.message_history)
        
        print("act_message", act_message)
        print("act_message content", act_message.content)
        parsed = self.action_parser.parse(act_message.content)
        print("parsed", parsed)
        
        response = parsed["response"].split(",")[0]
        command = parsed["command"].split(",")[0]
        add_object = parsed["add_object"].split(",")[0] 
        del_object = parsed["del_object"].split(",")[0] 
        
        
        self.message_history.append(act_message)
        return response, command, add_object, del_object
    def reset(self):
        self.message_history = [
            SystemMessage(content=self.docs),
        ]
        return
    
    
if __name__ == '__main__':

    with open("openai_api_key.txt") as fapi:
            api_key = fapi.read()
    env = SocialEnv()
    chat = ChatOpenAI(temperature=0, verbose=True, max_tokens=256, openai_api_key=api_key)
    agent = SocialBrain(model=chat, env=env)
    agent.reset()
   
    

    utterance_user = "Bring me a red cup"
    age = "elder"
    confidence_of_age = 90
    step = "set_parameters"
    interruptible= True
    move_arm = True
    move_arm = True
    move_base = True
    current_location = "null"
    destination_location = "null"
    objects_in_use = "null"
    dict_object = "{object_type: null, color: null, name: null, location: null, size: null}"
    

                   
    

    
    agent.information_process(utterance_user, age, confidence_of_age, step, interruptible, dict_object, move_arm, move_base, current_location, destination_location, objects_in_use)
    
    
    

        
    
    