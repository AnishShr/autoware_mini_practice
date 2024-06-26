#!/usr/bin/env python3

import rospy
import numpy as np
from sklearn.cluster import DBSCAN

from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured
from ros_numpy import numpify, msgify

from sensor_msgs.msg import PointCloud2

class PointsClusterer:
    def __init__(self):

        # Parameters
        self.cluster_min_size = rospy.get_param('~cluster_min_size')
        self.cluster_epsilon = rospy.get_param('~cluster_epsilon')
        self.output_frame = rospy.get_param('/detection/output_frame')        

        # Class variables        
        self.clusterer = DBSCAN(eps=self.cluster_epsilon,
                                min_samples=self.cluster_min_size)
        # Publishers
        self.points_cluster_pub = rospy.Publisher('points_clustered', PointCloud2, queue_size=1, tcp_nodelay=True)

        # Subscribers
        rospy.Subscriber('points_filtered', PointCloud2, self.points_callback, queue_size=1, buff_size=2**24, tcp_nodelay=True)

        rospy.loginfo("%s - initialized", rospy.get_name())


    def points_callback(self, msg):        
        data = numpify(msg)
        points = structured_to_unstructured(data[['x', 'y', 'z']], dtype=np.float32)
        labels = self.clusterer.fit_predict(points)

        assert points.shape[0] == labels.shape[0], "Point cloud points and labels are of different lengths"

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


    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('points_clusterer', log_level=rospy.INFO)
    node = PointsClusterer()
    node.run()
