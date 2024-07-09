#!/usr/bin/env python3

import rospy
from speech_processing.msg import *
from std_msgs.msg import String

import sys
import select
import gtts
from playsound import playsound

class SynthesizerNode:
    def __init__(self) -> None:
        self.sub_to_synthesizer = rospy.Subscriber("to_synthesizer", String, self.callback)
        self.pub_to_asr = rospy.Publisher("asr_activation", String, queue_size=10)

    def callback(self, data):
        rospy.loginfo(data)
        text = data.data
        if text:
            tts = gtts.gTTS(text)
            tts.save("system_transcript.mp3")
            self.pub_to_asr.publish("off")
            playsound("system_transcript.mp3")
            rospy.sleep(1)
            self.pub_to_asr.publish("on")
        

if __name__ == "__main__": 
    rospy.init_node('synthesizer_node', anonymous=True)
    
    speech_synthesis = SynthesizerNode()

    while not rospy.is_shutdown():
        rospy.spin()
