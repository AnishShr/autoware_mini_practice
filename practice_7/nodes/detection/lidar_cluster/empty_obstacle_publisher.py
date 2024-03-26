#!/usr/bin/env python3
import rospy
from autoware_msgs.msg import DetectedObjectArray

from sensor_msgs.msg import PointCloud2

class EmptyObstaclePublisher:
    def __init__(self):

        # Publish final objects
        self.empty_obstacle_pub = rospy.Publisher('final_objects', DetectedObjectArray, queue_size=1, tcp_nodelay=True)
        # Subscribe to pointcloud
        rospy.Subscriber('/lidar_center/points_raw', PointCloud2, self.pointcloud_callback, queue_size=1, buff_size=2**24, tcp_nodelay=True)

    def pointcloud_callback(self, msg):

        # Create empty DetectedObjectArray
        empty_objects = DetectedObjectArray()
        empty_objects.header = msg.header
        empty_objects.header.frame_id = msg.header.frame_id

        # Publish empty objects
        self.empty_obstacle_pub.publish(empty_objects)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('empty_obstacle_publisher', log_level=rospy.INFO)
    node = EmptyObstaclePublisher()
    node.run()