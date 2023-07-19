#!/usr/bin/env python3

import rospy
from speech_processing.msg import *
from std_msgs.msg import String

import sys
import select


class SynthesizerNode:
    def __init__(self) -> None:
        self.sub_to_synthesizer = rospy.Subscriber("to_synthesizer", String, self.callback)

    def callback(self, data):
        rospy.loginfo(data)


if __name__ == "__main__": 
    rospy.init_node('synthesizer_node', anonymous=True)
    
    speech_synthesis = SynthesizerNode()

    while not rospy.is_shutdown():
        rospy.sleep(0.1)
