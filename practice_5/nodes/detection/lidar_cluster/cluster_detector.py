#!/usr/bin/env python3

import rospy
import numpy as np

from shapely import MultiPoint
from tf2_ros import TransformListener, Buffer, TransformException
from numpy.lib.recfunctions import structured_to_unstructured
from ros_numpy import numpify, msgify

from sensor_msgs.msg import PointCloud2
from autoware_msgs.msg import DetectedObjectArray, DetectedObject
from std_msgs.msg import ColorRGBA, Header
from geometry_msgs.msg import Point32


BLUE80P = ColorRGBA(0.0, 0.0, 1.0, 0.8)

class ClusterDetector:
    def __init__(self):
        self.min_cluster_size = rospy.get_param('~min_cluster_size')
        self.output_frame = rospy.get_param('/detection/output_frame')
        self.transform_timeout = rospy.get_param('~transform_timeout')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

        self.objects_pub = rospy.Publisher('detected_objects', DetectedObjectArray, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('points_clustered', PointCloud2, self.cluster_callback, queue_size=1, buff_size=2**24, tcp_nodelay=True)

        rospy.loginfo("%s - initialized", rospy.get_name())


    def cluster_callback(self, msg):
        data = numpify(msg)
        points = structured_to_unstructured(data[['x', 'y', 'z']], dtype=np.float32)
        labels = structured_to_unstructured(data[['label']], dtype=np.int32)
        print(np.squeeze(labels))

        # print("---------------------Cluster callback-------------------------------")
        
        if msg.header.frame_id != self.output_frame:
            # print(f"points shape before transform: {points.shape}")
            # print(f"first row: {points[0]}")

            # Fetch transform
            try:
                transform = self.tf_buffer.lookup_transform(self.output_frame, msg.header.frame_id, msg.header.stamp, rospy.Duration(self.transform_timeout))
                tf_matrix = numpify(transform.transform).astype(np.float32)
                # make copy of points
                points = points.copy()
                # turn into homogeneous coordinates
                points = np.column_stack((points, np.array(np.ones(points.shape[0]))))                
                # transform points to target frame
                points = points.dot(tf_matrix.T)
                
                # print("tf matrix:")
                # print(tf_matrix)
                # print(f"points shape after tranform: {points.shape}")
                # print(f"first row: {points[0]}")

                detected_object_array = []

                if len(labels) == 0:
                    object_array = DetectedObjectArray()

                    object_array.header.stamp = msg.header.stamp
                    object_array.header.frame_id = self.output_frame

                    object_array.objects = detected_object_array

                    self.objects_pub.publish(object_array)

                else:

                    for i in range(np.max(labels)+1):
                        
                        mask = labels == i                    
                        mask = np.squeeze(mask)

                        if np.sum(mask) >= self.min_cluster_size:

                            points3d = points[mask, :3]

                            # Centroid
                            centroid_x, centroid_y, centroid_z = np.mean(points3d, axis=0)

                            # convex hull
                            points2d = MultiPoint(points[mask, :2])
                            hull = points2d.convex_hull
                            convex_hull_points = [Point32(x, y, centroid_z) for x, y in hull.exterior.coords]
                            
                            object = DetectedObject()

                            object.label = "Unknown"
                            object.color = BLUE80P
                            object.valid = True
                            object.space_frame = self.output_frame
                            object.pose_reliable = True
                            object.velocity_reliable = False
                            object.acceleration_reliable = False
                            
                            object.header.stamp = msg.header.stamp
                            object.header.frame_id = self.output_frame
                            object.id = i

                            object.pose.position.x = centroid_x
                            object.pose.position.y = centroid_y
                            object.pose.position.z = centroid_z

                            object.convex_hull = convex_hull_points

                            detected_object_array.append(object)
                
                    object_array = DetectedObjectArray()

                    object_array.header.stamp = msg.header.stamp
                    object_array.header.frame_id = self.output_frame

                    object_array.objects = detected_object_array

                    self.objects_pub.publish(object_array)


            except (TransformException, rospy.ROSTimeMovedBackwardsException) as e:
                rospy.logwarn("%s - %s", rospy.get_name(), e)
                return
        
        



    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('cluster_detector', log_level=rospy.INFO)
    node = ClusterDetector()
    node.run()
