#!/usr/bin/env python3

import rospy
from speech_processing.msg import *
from std_msgs.msg import String
from dialogue_system.dialogue_dummy import dialogue_system

import sys
import select


class dialogue_node_class:
    def __init__(self) -> None:
        # self.pub = rospy.Publisher('to_robot', message_to_robot, queue_size=10)
        self.pub_interrupt_minor = rospy.Publisher('robot_minor_interruption', message_to_robot, queue_size=10)
        self.pub_interrupt_major = rospy.Publisher('robot_major_interruption', message_to_robot, queue_size=10)

        self.pub_to_synthesizer = rospy.Publisher('to_synthesizer', String, queue_size=10)

        self.dia = dialogue_system()
        
    def callback(self, data):
    
        rospy.loginfo(rospy.get_caller_id() + "I heard %s %s %i %f ", data.transcript, data.age, data.confidence)
        
        response = self.dia.process_speech_input(data.transcript, data.age, data.confidence)
        if response is not None:
            # self.pub.publish(*response)
            self.pub_interrupt_minor.publish(*response)
            
    def callback_from_robot(self, data):
        response_to_synthesizer = self.dia.process_robot_input(data.step, data.interruptable, data.object_info,
                                                               data.move_arm, data.move_base, data.current_location,
                                                               data.destination_location)
        self.pub_to_synthesizer.publish(response_to_synthesizer)

    def listen_and_publish(self):
        self.sub_speech = rospy.Subscriber("speech_publisher", command_and_age, self.callback)
        self.sub_from_robot = rospy.Subscriber("from_robot", message_from_robot, self.callback_from_robot)

        #rospy.spin()
        
    def user_loop(self):
        while not rospy.is_shutdown():
        
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                user_input = sys.stdin.readline().rstrip()
                if user_input[0] == "#":
                    #command not for communication 
                    print(user_input[1:])
                    if user_input[1:] == "get_robot_states":
                        print(self.dia.get_robot_states())
                    else:
                        print("unknown command")

                else: 
                    print(f"Inputted Command {user_input}")
                    minor_major ,response = self.dia.process_commandline_input(user_input)
                    if minor_major == "major":
                        self.pub_interrupt_major.publish(*response) 
                    elif minor_major == "minor":
                        self.pub_interrupt_minor.publish(*response) 
            else:
                
                rospy.sleep(0.1)


if __name__ == "__main__": 
    rospy.init_node('dialogoue_node', anonymous=True)
    
    dia = dialogue_node_class()
    dia.listen_and_publish()
    dia.user_loop()