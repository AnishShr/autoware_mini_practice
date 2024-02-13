import rospy
from std_msgs.msg import String

rospy.init_node('publisher')

message = rospy.get_param('~message', 'Hello World!')
rate = rospy.Rate(rospy.get_param('~rate', '5'))

pub = rospy.Publisher('/message', String, queue_size=10)

while not rospy.is_shutdown():
    pub.publish(message)
    rate.sleep()