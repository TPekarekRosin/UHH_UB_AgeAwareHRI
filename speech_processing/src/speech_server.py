
import rospy
from speech_processing.srv import SpeechRecognition


def speech_recognized(req):
    signal = req.signal
    # TODO add speech recognition
    logits = ((0.1, 0.2, 0.3), (0.4, 0.5, 0.6))
    age_estimation = 0.6
    return SpeechRecognition(logits, age_estimation)


def speech_server():
    rospy.init_node('speech_server', anonymous=True)
    rospy.Service('speech_recognized', SpeechRecognition, speech_recognized)
    print("\nReady to recognize speech.")
    rospy.spin()

