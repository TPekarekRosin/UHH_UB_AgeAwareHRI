#!/usr/bin/env python3

import rospy
from speech_processing.msg import *
from std_msgs.msg import String


class robot:
    def __init__(self) -> None:
        self.pub = rospy.Publisher('from_robot', message_from_robot, queue_size=10)
        pass

    def callback_minor(self, data):
        rospy.loginfo(f"Minor interruption of type {data.command}")
        if data.command == "object":
            response = message_from_robot()
            response.step = 'transporting.search'
            response.interruptable = True
            response.object.append(dict_object())
            response.object[0].type = 'cup'
            response.object[0].color = 'red'
            response.object[0].name = ''
            response.object[0].location = ''
            response.object[0].size = ''
            response.move_arm = True
            response.move_base = True
            response.current_location = 'kitchen'
            response.destination_location = 'kitchen'
            self.pub.publish(response)
        elif data.command == "action":
            response = message_from_robot()
            response.step = 'transporting.fetch'
            response.interruptable = False
            response.object.append(dict_object())
            response.object[0].type = 'cup'
            response.object[0].color = 'red'
            response.object[0].name = ''
            response.object[0].location = ''
            response.object[0].size = ''
            response.move_arm = True
            response.move_base = False
            response.current_location = 'kitchen'
            response.destination_location = 'kitchen'
            self.pub.publish(response)

    def callback_major(self, data):
        rospy.loginfo(f"Major interruption of type {data.command}")
        if data.command == "stop":
            #robot.force_stop()
            response = message_from_robot()
            response.step = 'idle'
            response.interruptable = False
            response.object.append(dict_object())
            response.object[0].type = ''
            response.object[0].color = ''
            response.object[0].name = ''
            response.object[0].location = ''
            response.object[0].size = ''
            response.move_arm = False
            response.move_base = False
            response.current_location = 'kitchen'
            response.destination_location = 'kitchen'
            self.pub.publish(response)

    def listen(self):
        rospy.init_node('robot_node', anonymous=True)
        
        self.sub = rospy.Subscriber("robot_minor_interruption", message_to_robot, self.callback_minor)
        self.sub = rospy.Subscriber("robot_major_interruption", message_to_robot, self.callback_major)

    def do_dummy_stuff(self):
        rospy.spin()    


if __name__ == "__main__": 
    robot = robot()
    robot.listen()
    robot.do_dummy_stuff()