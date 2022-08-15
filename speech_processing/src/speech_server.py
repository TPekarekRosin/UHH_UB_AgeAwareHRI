#!/usr/bin/env python3

import rospy
from speech_processing.srv import *
from age_recognition.model_components.live_inference import LiveInference


def speech_recognized(req):
    signal = req.signal

    inference_model = LiveInference()
    recognized, age_estimation = inference_model.buffer_to_text(signal)

    return SpeechRecognitionResponse(recognized, age_estimation)


def speech_server():
    rospy.init_node('speech_server', anonymous=True)
    rospy.Service('speech_recognized', SpeechRecognition, speech_recognized)
    print("\nReady to recognize speech.")
    rospy.spin()


if __name__ == "__main__":
    speech_server()

