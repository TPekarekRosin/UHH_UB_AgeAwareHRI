#!/usr/bin/env python3

import rospy
from speech_processing.msg import *
from std_msgs.msg import String

import sys
import select

class synthesizer_node_class:
    def __init__(self) -> None:
        #initiate synthesizer when existing
        pass
    def callback(self,data):
        print(data)

        rospy.loginfo(rospy.get_caller_id() + f"I heard {data} ", )
        

    def create_subscriber(self):
        self.sub = rospy.Subscriber("to_synthesizer", String, self.callback)

        
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
    rospy.init_node('synthesizer_node', anonymous=True)
    
    dia = synthesizer_node_class()
    dia.create_subscriber()
    dia.user_loop()