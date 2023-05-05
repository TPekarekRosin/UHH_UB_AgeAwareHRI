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
            self.pub.publish("Stopped", False) 
            print("stopped")

        #rospy.loginfo(rospy.get_caller_id() + "I heard %s %s %i %f ", data.text,data.command,data.age,data.confidence)
        

    def listen(self):
        rospy.init_node('robot_node', anonymous=True)
        self.sub = rospy.Subscriber("to_robot", message_to_robot, self.callback)


        
    def do_dummy_stuff(self):
        rospy.spin()    


if __name__ == "__main__": 
    robot = robot()
    robot.listen()
    robot.do_dummy_stuff()