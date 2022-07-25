#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import String
import random
from age_recognition_pub.msg import age_recognition

def talker():
    pub = rospy.Publisher('age_recognition', age_recognition)
    rospy.init_node('custom_age_talker', anonymous=True)
    r = rospy.Rate(10) #10hz
    msg = age_recognition()
    msg.command = "bring me coffe"
    msg.age = random.randint(0,1)

    while not rospy.is_shutdown():
        rospy.loginfo(msg)
        pub.publish(msg)
        r.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException: pass
