#!/usr/bin/env python3

import rospy
from speech_processing.msg import *
from std_msgs.msg import String
from dialogue_system.dialogue_dummy import dialogue_system

import sys
import select

class dialogue_node_class:
    def __init__(self) -> None:
        self.pub = rospy.Publisher('to_robot', message_to_robot, queue_size=10)
        self.dia= dialogue_system()
    def callback(self,data):
        print(data)
        
        #data.text
        #data.command
        #data.age
        #data.confidence
        rospy.loginfo(rospy.get_caller_id() + "I heard %s %s %i %f ", data.text,data.command,data.age,data.confidence)
        
        response = self.dia.process_speech_input(data.text,data.command,data.age, data.confidence)
        if response != None:
            self.pub.publish(*response)
    def callback_from_robot(self,data):
        self.dia.process_robot_input(robot_interruptable=data.interruptable, robot_state= data.step)

    def listen_and_publish(self):
        self.sub_speech = rospy.Subscriber("speech_publisher", command_and_age, self.callback)
        self.sub_from_robot = rospy.Subscriber("from_robot", message_from_robot, self.callback_from_robot)

        
        #rospy.spin()
    def user_loop(self):
        while not rospy.is_shutdown():
        
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                user_input = sys.stdin.readline().rstrip()
                
                print(f"Inputted Command {user_input}")
                response = self.dia.process_commandline_input()
            else:
                
                rospy.sleep(0.1)

if __name__ == "__main__": 
    rospy.init_node('dialogoue_node', anonymous=True)
    
    dia = dialogue_node_class()
    dia.listen_and_publish()
    dia.user_loop()