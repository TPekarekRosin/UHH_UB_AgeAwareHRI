import gtts
from playsound import playsound
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain.output_parsers import RegexParser

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
        self.env = env
        self.docs = self.get_docs(env)
        self.action_parser = RegexParser( 
            regex=r"^action: (.*)object: (.*)color: (.*)size: (.*)generation: (.*)",
            output_keys=["action", "object", "color", "size", "generation"],
            # output_keys=["action", "object"],
            default_output_key="action",
        )
        
        self.message_history = []
        # self.ret = 0
        
    def information_process(self):
        utterance_user, confidence_text, age, confidence_age = self.get_user_utterance_and_age()
        
        robot_status_current = self.get_robot_status()
        step = robot_status_current["step"]
        interruptible = robot_status_current["interruptible"]
        human_message = f" utterance: {utterance_user}, \n step: {step}, \n interruptible: {interruptible}\n"
        print("human_message", human_message)
        self.message_history.append(HumanMessage(content=human_message)
        )
        act_message = self.model(self.message_history)
        print("act_message", act_message)
        parsed = self.action_parser.parse(act_message.content)
        print("parsed", parsed)
        action = parsed["action"].split(",")[0]
        object = parsed["object"].split(",")[0]
        color = parsed["color"].split(",")[0] 
        size = parsed["size"].split(",")[0] 
        generation = parsed["generation"]
        
        
        self.message_history.append(act_message)
        return age, action, object, color, size, generation
    
    def get_user_utterance_and_age(self):
        # how to subscribe topic
        
        utterance = input('user:')
        confidence_text = 1
        age = "young"
        confidence_age = 0.9
        return utterance, confidence_text, age, confidence_age
    
    ##### get message from robot #####
    # string step
    # bool interruptable
    # dict object {  
    #               string type,
    #               string color,
    #               string name,
    #               string location,
    #               string size
    #              }
    # bool move_arm
    # bool move_base
    # string current_location
    # string destination_location
    def get_robot_status(self):
        
        # question: how to subscribte topic 
        step = input('step:')
        interruptible = input('interruptible:')
        object_dict = input('object_dict:')
        move_base = input('move_base:')
        move_arm = input('move_arm:')
        current_location = input('current_location:')
        destination_location = input('destination_location:')
        
        
        robot_status = {
            "step": step,
            "interruptible": interruptible
        }
        return robot_status
    def get_objects_in_use(self):
        objects_in_use = ["milk", "bowl"]
        return objects_in_use
    
    def reset(self):
        self.message_history = [
            SystemMessage(content=self.docs),
            # SystemMessage(content=self.instructions),
        ]
        return
    
    def text_to_speech(self, name):
        text = "i will search for the"+ name
        tts = gtts.gTTS(text)
        tts.save(name +".mp3")
        playsound(name +".mp3")
        return True
    
if __name__ == '__main__':

    
    socialbrain = SocialBrain()
    
    socialbrain.text_to_speech("red cup")
    
    
    
    

        
    
    