#!/usr/bin/env python3

import rospy
import numpy as np
from sklearn.cluster import DBSCAN

# from shapely import MultiPoint
# from tf2_ros import TransformListener, Buffer, TransformException
from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured
from ros_numpy import numpify, msgify

from sensor_msgs.msg import PointCloud2
# from autoware_msgs.msg import DetectedObjectArray, DetectedObject
# from std_msgs.msg import ColorRGBA, Header
# from geometry_msgs.msg import Point32


# BLUE80P = ColorRGBA(0.0, 0.0, 1.0, 0.8)

class PointsClusterer:
    def __init__(self):

        # Parameters
        self.cluster_min_size = rospy.get_param('~cluster_min_size')
        self.cluster_epsilon = rospy.get_param('~cluster_epsilon')
        self.output_frame = rospy.get_param('/detection/output_frame')        

        # Class variables        
        self.clusterer = DBSCAN(eps=self.cluster_epsilon,
                                min_samples=self.cluster_min_size)
        # Publihsers
        self.points_cluster_pub = rospy.Publisher('points_clustered', PointCloud2, queue_size=1, tcp_nodelay=True)

        # Subscribers
        rospy.Subscriber('points_filtered', PointCloud2, self.points_callback, queue_size=1, buff_size=2**24, tcp_nodelay=True)

        rospy.loginfo("%s - initialized", rospy.get_name())


    def points_callback(self, msg):        
        data = numpify(msg)
        points = structured_to_unstructured(data[['x', 'y', 'z']], dtype=np.float32)
        labels = self.clusterer.fit_predict(points)

        valid_points_indices = labels >= 0
        valid_points = points[valid_points_indices]
        valid_lables = labels[valid_points_indices]

        points_labelled = np.column_stack((valid_points, valid_lables))
        
        # convert labelled points to PointCloud2 format
        data = unstructured_to_structured(points_labelled, dtype=np.dtype([
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('label', np.int32)
        ]))

        # publish clustered points message
        cluster_msg = msgify(PointCloud2, data)
        cluster_msg.header.stamp = msg.header.stamp
        cluster_msg.header.frame_id = msg.header.frame_id

        self.points_cluster_pub.publish(cluster_msg)
        # print(cluster_msg.header)


    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('points_clusterer', log_level=rospy.INFO)
    node = PointsClusterer()
    node.run()
