#!/usr/bin/env python3

import rospy
from speech_processing.msg import *
from speech_processing.src.dialogue_system.social_brain import SocialBrain
from speech_processing.src.dialogue_system.social_env import SocialEnv
from std_msgs.msg import String
from dialogue_system.dialogue_system import DialogueSystem

import sys
import select

from langchain.chat_models import ChatOpenAI


class DialogueNode:
    def __init__(self) -> None:
        # self.pub = rospy.Publisher('to_robot', message_to_robot, queue_size=10)
        self.pub_interrupt_minor = rospy.Publisher('robot_minor_interruption', message_to_robot, queue_size=10)
        self.pub_interrupt_major = rospy.Publisher('robot_major_interruption', message_to_robot, queue_size=10)

        self.pub_to_synthesizer = rospy.Publisher('to_synthesizer', String, queue_size=10)

        self.sub_speech = rospy.Subscriber("speech_publisher", command_and_age, self.callback_from_asr)
        self.sub_from_robot = rospy.Subscriber("from_robot", message_from_robot, self.callback_from_robot)

        self.dialogue_system = DialogueSystem()
        
        with open("openai_api_key.txt") as fapi:
            self.api_key = fapi.read()
        self.env = SocialEnv()
        self.chat = ChatOpenAI(temperature=0, verbose=True, max_tokens=256, openai_api_key=self.api_key)
        self.agent = SocialBrain(model=self.chat, env=self.env)
        
    def callback_from_asr(self, data):
        rospy.loginfo(data)

        minor_or_major, response = self.dialogue_system.process_speech_input(data.transcript,
                                                                             data.age,
                                                                             data.confidence)
        print(minor_or_major, response)
        if minor_or_major == 'minor':
            # self.pub.publish(*response)
            self.pub_interrupt_minor.publish(response)
        elif minor_or_major == 'major':
            self.pub_interrupt_major.publish(response)
        else:
            print('Error')

    def callback_from_robot(self, data):
        rospy.loginfo(data)
        response_to_synthesizer = self.dialogue_system.process_robot_input(data.step, data.interruptable, data.object,
                                                               data.move_arm, data.move_base, data.current_location,
                                                               data.destination_location)
        print(response_to_synthesizer)
        self.pub_to_synthesizer.publish(response_to_synthesizer)


if __name__ == "__main__": 
    rospy.init_node('dialogoue_node', anonymous=True)
    
    dialogue_node = DialogueNode()
    while not rospy.is_shutdown():
        rospy.sleep(0.1)
