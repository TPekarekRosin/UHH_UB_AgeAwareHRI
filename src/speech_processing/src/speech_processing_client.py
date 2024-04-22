#!/usr/bin/env python3
import rospy
import pyaudio as pa
from std_msgs.msg import String
from speech_processing.msg import *
from age_recognition.live_model import ASRLiveModel
import sounddevice as sd
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def speech_publisher(transcript, age, confidence):
    pub = rospy.Publisher('speech_publisher', command_and_age, queue_size=10)
    # rospy.init_node('custom_age_talker', anonymous=True)
    r = rospy.Rate(10)  # 10hz

    msg = command_and_age()
    msg.transcript = transcript
    msg.age = age
    msg.confidence = confidence

    #while not rospy.is_shutdown():
    rospy.loginfo(msg)
    pub.publish(msg)
    r.sleep()


if __name__ == "__main__":
    rospy.init_node('speech_engine', anonymous=True)

    print(sd.query_devices())

    # choose audio device
    p = pa.PyAudio()
    print('Available audio input devices:')
    input_devices = []
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev.get('maxInputChannels'):
            input_devices.append(i)
            print(i, dev.get('name'))

    if len(input_devices):
        dev_idx = -2
        while dev_idx not in input_devices:
            print('Please type input device ID:')
            dev_idx = int(input())

    device = p.get_device_info_by_index(dev_idx)
    asr = ASRLiveModel(device.get('name'))
    sub_speech = rospy.Subscriber("asr_activation", String, asr.callback)
    asr.start()

    """while not rospy.is_shutdown():
        
        transcript, confidence, age_estimation = asr.get_last_text()
        print(f"\tAge Estimation: {age_estimation.item()}"
              + f"\tHeard: '{transcript}'")"""
