#!/usr/bin/env python3
import sys
import rospy
from speech_processing.srv import *
from speech_processing.msg import *


def age_recognition_publisher(command, age):
    pub = rospy.Publisher('age_recognition', command_and_age)
    rospy.init_node('custom_age_talker', anonymous=True)
    r = rospy.Rate(10)  # 10hz

    msg = command_and_age()
    msg.command = command
    msg.age = age

    while not rospy.is_shutdown():
        rospy.loginfo(msg)
        pub.publish(msg)
        r.sleep()


def speech_recognized_client(signal):
    rospy.wait_for_service('speech_recognized')
    try:
        recognize_speech = rospy.ServiceProxy('speech_recognized', SpeechRecognition)
        response = recognize_speech(signal)
        return response.recognized, response.age_estimation
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)
