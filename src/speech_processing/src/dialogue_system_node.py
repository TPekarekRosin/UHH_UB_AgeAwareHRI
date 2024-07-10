#!/usr/bin/env python3

import rospy
from speech_processing.msg import *
from std_msgs.msg import String
from dialogue_system.dialogue_system import DialogueSystem
import gtts
from playsound import playsound


import sys
import select


class DialogueNode:
    def __init__(self) -> None:
        # self.pub = rospy.Publisher('to_robot', message_to_robot, queue_size=10)
        self.pub_interrupt_minor = rospy.Publisher('robot_minor_interruption', message_to_robot, queue_size=10)
        self.pub_interrupt_major = rospy.Publisher('robot_major_interruption', message_to_robot, queue_size=10)

        self.pub_to_synthesizer = rospy.Publisher('to_synthesizer', String, queue_size=10)
        # publisher to activate and deactivate speech recognition
        self.pub_to_speech = rospy.Publisher('asr_activation', String, queue_size=10)
        
        self.sub_speech = rospy.Subscriber("speech_publisher", command_and_age, self.callback_from_asr)
        self.sub_from_robot = rospy.Subscriber("from_robot", message_from_robot, self.callback_from_robot)

        self.sub_objects_in_use = rospy.Subscriber('objects_in_use', message_objects_in_use, self.callback_obj_in_use)

        self.dialogue_system = DialogueSystem()

    def callback_obj_in_use(self, data):
        # print("objects in use")
        # rospy.loginfo(data)
        length = len(data.objects)
        self.dialogue_system.objects_in_use.clear()
        for i in range(length):    
            type = data.objects[i].type
            color = data.objects[i].color
            name = data.objects[i].name
            location = data.objects[i].location
            size = data.objects[i].size
            current_object = {'type': type,
                                'color': color,
                                'name': name,
                                'location': location,
                                'size': size,              
                            }
            self.dialogue_system.objects_in_use.append(current_object)
              
    def callback_from_asr(self, data):
        rospy.loginfo(data)
        self.dialogue_system.user_data["transcript"] = data.transcript
        self.dialogue_system.user_data["age"] = data.age
        self.dialogue_system.user_data["confidence"] = data.confidence
        # print("CONTROL PRINT", self.dialogue_system.user_data)

        minor_or_major, response_to_robot, system_transcript= self.dialogue_system.process_speech_input(data.transcript,
                                                                             data.age,
                                                                             data.confidence)
        # self.text_to_speech(system_transcript)
        self.pub_to_synthesizer.publish(system_transcript)
        print("minor_or_major", minor_or_major)
        if minor_or_major == 'minor':
            # response is a message to robot
            response = message_to_robot()
            response.age = data.age
            response.confidence = data.confidence
            if not response_to_robot:
                response.command = ''
                # properties of added object
                response.add_object.append(dict_object())
                response.add_object[0].type = ''
                response.add_object[0].color = ''
                response.add_object[0].name = ''
                response.add_object[0].location = ''
                response.add_object[0].size = ''
                # properties of deleted object
                response.del_object.append(dict_object())
                response.del_object[0].type = ''
                response.del_object[0].color = ''
                response.del_object[0].name = ''
                response.del_object[0].location = ''
                response.del_object[0].size = ''
            else:
                response.command = response_to_robot["command"]
                response.age = data.age 
                response.confidence = data.confidence
                # properties of added object
                response.add_object.append(dict_object())
                response.add_object[0].type = response_to_robot["add_type"]
                response.add_object[0].color = response_to_robot["add_color"]
                response.add_object[0].name = response_to_robot["add_name"]
                response.add_object[0].location = response_to_robot["add_location"]
                response.add_object[0].size = response_to_robot["add_size"]
                # properties of deleted object
                response.del_object.append(dict_object())
                response.del_object[0].type = response_to_robot["del_type"]
                response.del_object[0].color = response_to_robot["del_color"]
                response.del_object[0].name = response_to_robot["del_name"]
                response.del_object[0].location = response_to_robot["del_location"]
                response.del_object[0].size = response_to_robot["del_size"]
            self.pub_interrupt_minor.publish(response)
            print('now in the minor interruption')
        elif minor_or_major == 'major':
            response = message_to_robot()
            response.command = 'stop'
            response.age = data.age
            response.confidence = data.confidence
            # # properties of added object
            # response.add_object.append(dict_object())
            # response.add_object[0].type = response_to_robot["add_type"]
            # response.add_object[0].color = response_to_robot["add_color"]
            # response.add_object[0].name = response_to_robot["add_name"]
            # response.add_object[0].location = response_to_robot["add_location"]
            # response.add_object[0].size = response_to_robot["add_size"]
            # # properties of deleted object
            # response.del_object.append(dict_object())
            # response.del_object[0].type = response_to_robot["del_type"]
            # response.del_object[0].color = response_to_robot["del_color"]
            # response.del_object[0].name = response_to_robot["del_name"]
            # response.del_object[0].location = response_to_robot["del_location"]
            # response.del_object[0].size = response_to_robot["del_size"]
            self.pub_interrupt_major.publish(response)
            print('now in the major interruption')
        else:
            print('Error')

    def callback_from_robot(self, data):
        # rospy.loginfo(data)
        
        self.dialogue_system.robot_data["step"] = data.step
        self.dialogue_system.robot_data["interruptable"] = data.interruptable
        self.dialogue_system.robot_data["object"] = data.object
        self.dialogue_system.robot_data["move_arm"] = data.move_arm
        self.dialogue_system.robot_data["move_base"] = data.move_base
        self.dialogue_system.robot_data["current_location"] = data.current_location                                                        
        self.dialogue_system.robot_data["destination"] = data.destination_location

        system_transcript = self.dialogue_system.process_robot_input(data.step, data.interruptable, data.object,
                                                               data.move_arm, data.move_base, data.current_location,
                                                               data.destination_location)
        print("system_transcript", system_transcript)
        # self.text_to_speech(response_to_synthesizer)
        self.pub_to_synthesizer.publish(system_transcript)
        
    def text_to_speech(self, text):
        # tts = gtts.gTTS(text)
        # tts.save("system_transcript.mp3")
        # playsound("system_transcript.mp3")
        print(text)

    

if __name__ == "__main__": 
    rospy.init_node('dialogoue_node', anonymous=True)
    
    dialogue_node = DialogueNode()
    while not rospy.is_shutdown():
        rospy.sleep(0.1)
