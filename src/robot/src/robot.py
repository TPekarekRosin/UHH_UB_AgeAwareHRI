#!/usr/bin/env python3

import rospy
from speech_processing.msg import *
from std_msgs.msg import String


class robot:
    def __init__(self) -> None:
        self.pub_from_robot = rospy.Publisher('from_robot', message_from_robot, queue_size=10)
        self.pub_in_use = rospy.Publisher('objects_in_use', message_objects_in_use, queue_size=10)
        rospy.set_param('object_in_use', 'cup,bowl,spoon,cornflakes')
        self.from_robot = message_from_robot()
        self.obj_in_use = message_objects_in_use()
        self.from_robot.step = 'idle'
        self.from_robot.interruptable = False
        self.from_robot.object.append(dict_object())
        self.from_robot.object[0].type = ''
        self.from_robot.object[0].color = ''
        self.from_robot.object[0].name = ''
        self.from_robot.object[0].location = ''
        self.from_robot.object[0].size = ''
        self.from_robot.move_arm = False
        self.from_robot.move_base = False
        self.from_robot.current_location = 'kitchen'
        self.from_robot.destination_location = 'kitchen'

        self.obj_in_use.objects.append(dict_object())
        self.obj_in_use.objects[0].type = ''
        self.obj_in_use.objects[0].color = ''
        self.obj_in_use.objects[0].name = ''
        self.obj_in_use.objects[0].location = ''
        self.obj_in_use.objects[0].size = ''
        pass

    def from_robot_pub(self):
        while not rospy.is_shutdown():
            self.pub_from_robot.publish(self.from_robot)
            self.pub_in_use.publish(self.obj_in_use)
            self.rate.sleep()

    def callback_minor(self, data):
        rospy.loginfo(f"Minor interruption of type {data.command}")
        if data.command == "object":
            self.from_robot.step = 'transporting.search'
            self.from_robot.interruptable = True
            self.from_robot.object[0].type = 'cup'
            self.from_robot.object[0].color = 'red'
            self.from_robot.object[0].name = ''
            self.from_robot.object[0].location = ''
            self.from_robot.object[0].size = ''
            self.from_robot.move_arm = True
            self.from_robot.move_base = True
            self.from_robot.current_location = 'kitchen'
            self.from_robot.destination_location = 'kitchen'

            self.obj_in_use.objects[0].type = 'cup'
            self.obj_in_use.objects[0].color = 'red'
            self.obj_in_use.objects[0].name = ''
            self.obj_in_use.objects[0].location = ''
            self.obj_in_use.objects[0].size = ''

        elif data.command == "action":
            self.from_robot.step = 'transporting.fetch'
            self.from_robot.interruptable = False
            self.from_robot.object[0].type = 'cup'
            self.from_robot.object[0].color = 'red'
            self.from_robot.object[0].name = ''
            self.from_robot.object[0].location = ''
            self.from_robot.object[0].size = ''
            self.from_robot.move_arm = True
            self.from_robot.move_base = False
            self.from_robot.current_location = 'kitchen'
            self.from_robot.destination_location = 'kitchen'

            self.obj_in_use.objects[0].type = 'cup'
            self.obj_in_use.objects[0].color = 'red'
            self.obj_in_use.objects[0].name = ''
            self.obj_in_use.objects[0].location = ''
            self.obj_in_use.objects[0].size = ''

    def callback_major(self, data):
        rospy.loginfo(f"Major interruption of type {data.command}")
        if data.command == "stop":
            # robot.force_stop()
            self.from_robot.step = 'idle'
            self.from_robot.interruptable = False
            self.from_robot.object[0].type = ''
            self.from_robot.object[0].color = ''
            self.from_robot.object[0].name = ''
            self.from_robot.object[0].location = ''
            self.from_robot.object[0].size = ''
            self.from_robot.move_arm = False
            self.from_robot.move_base = False
            self.from_robot.current_location = 'kitchen'
            self.from_robot.destination_location = 'kitchen'

            self.obj_in_use.objects[0].type = ''
            self.obj_in_use.objects[0].color = ''
            self.obj_in_use.objects[0].name = ''
            self.obj_in_use.objects[0].location = ''
            self.obj_in_use.objects[0].size = ''

    def listen(self):
        rospy.init_node('robot_node', anonymous=True)
        self.rate = rospy.Rate(10)  # 10hz
        self.sub = rospy.Subscriber("robot_minor_interruption", message_to_robot, self.callback_minor)
        self.sub = rospy.Subscriber("robot_major_interruption", message_to_robot, self.callback_major)

    def do_dummy_stuff(self):
        rospy.spin()


if __name__ == "__main__":
    robot = robot()
    robot.listen()
    robot.from_robot_pub()
    robot.do_dummy_stuff()
