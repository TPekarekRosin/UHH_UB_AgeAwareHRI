#!/usr/bin/env python

from __future__ import print_function

import sys
import rospy
from speech_processing.srv import SpeechRecognition
from speech_processing.msg import command_and_age
"""
speech_client.py handles server calls to the speech server, and passes the response
along to the decoder, afterwards the decoded speech and binary age is published
"""


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
        recognize_speech = rospy.ServiceProxy('add_two_ints', SpeechRecognition)
        response = recognize_speech(signal)
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

    # TODO: decode the logits and age_estimation
    command = "bring me coffee"
    age = 0
    try:
        age_recognition_publisher(command, age)
    except rospy.ROSInterruptException:
        pass
