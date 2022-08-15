
import rospy
from speech_processing.srv import *
from age_recognition.model_components.live_model import live_speech_recognition


def speech_recognized(req):
    signal = req.signal
    # recognized, age_estimation = live_speech_recognition(signal)
    recognized = "bring me coffee"
    age_estimation = 0.6
    return SpeechRecognitionResponse(recognized, age_estimation)


def speech_server():
    rospy.init_node('speech_server', anonymous=True)
    rospy.Service('speech_recognized', SpeechRecognition, speech_recognized)
    print("\nReady to recognize speech.")
    rospy.spin()

