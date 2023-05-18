#!/usr/bin/env python3

import rospy
from speech_processing.msg import *
from std_msgs.msg import String


class robot:
    def __init__(self) -> None:
        self.pub = rospy.Publisher('from_robot', message_from_robot, queue_size=10)
        pass
    def callback(self,data):
        print(data)
        print(data.command)
        if data.command == "stop":
            print("starting the stopping")
            self.pub.publish("stopping", False)
            rospy.sleep(10)
            self.pub.publish("Idle", False) 
            print("stopped")

        #rospy.loginfo(rospy.get_caller_id() + "I heard %s %s %i %f ", data.text,data.command,data.age,data.confidence)
        
    def callback_minor(self,data):
        rospy.loginfo(f"Minor interruption of type {data.command}")
        if data.command == "continue":
            self.pub.publish("working", True)
            #actualrobot.do_something()


    def callback_major(self,data):
        rospy.loginfo(f"Major interruption of type {data.command}")
        if data.command == "stop":
            #robot.force_stop()
            self.pub.publish("Idle", False) 
            


    def listen(self):
        rospy.init_node('robot_node', anonymous=True)
        self.sub = rospy.Subscriber("to_robot", message_to_robot, self.callback)
        self.sub = rospy.Subscriber("robot_minor_interruption", message_to_robot, self.callback_minor)
        self.sub = rospy.Subscriber("robot_major_interruption", message_to_robot, self.callback_major)



        
    def do_dummy_stuff(self):
        rospy.spin()    


if __name__ == "__main__": 
    robot = robot()
    robot.listen()
    robot.do_dummy_stuff()