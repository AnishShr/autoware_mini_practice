#!/usr/bin/env python3

import math
import rospy
import numpy as np

from tf.transformations import quaternion_from_euler
from tf2_ros import TransformBroadcaster
from pyproj import CRS, Transformer, Proj

from novatel_oem7_msgs.msg import INSPVA
from geometry_msgs.msg import PoseStamped, TwistStamped, Quaternion, TransformStamped

# convert azimuth to yaw angle
def convert_azimuth_to_yaw(azimuth):
    """
    Converts azimuth to yaw. Azimuth is CW angle from the North. Yaw is CCW angle from the East.
    :param azimuth: azimuth in radians
    :return: yaw in radians
    """
    yaw = -azimuth + math.pi/2
    # Clamp within 0 to 2 pi
    if yaw > 2 * math.pi:
        yaw = yaw - 2 * math.pi
    elif yaw < 0:
        yaw += 2 * math.pi

    return yaw

class Localizer:
    def __init__(self):

        # Parameters
        self.undulation = rospy.get_param('/undulation')
        utm_origin_lat = rospy.get_param('/utm_origin_lat')
        utm_origin_lon = rospy.get_param('/utm_origin_lon')

        # Internal variables
        self.crs_wgs84 = CRS.from_epsg(4326)
        self.crs_utm = CRS.from_epsg(25835)
        self.utm_projection = Proj(self.crs_utm)

        # Subscribers
        rospy.Subscriber('/novatel/oem7/inspva', INSPVA, self.transform_coordinates)

        # Publishers
        self.current_pose_pub = rospy.Publisher('current_pose', PoseStamped, queue_size=10)
        self.current_velocity_pub = rospy.Publisher('current_velocity', TwistStamped, queue_size=10)
        self.br = TransformBroadcaster()

        # Create coordinate transformer
        self.transformer = Transformer.from_crs(self.crs_wgs84, self.crs_utm)
        self.origin_x, self.origin_y = self.transformer.transform(utm_origin_lat, utm_origin_lon)

        # Create PoseStamped msg to publish current_pose
        self.current_pose_msg = PoseStamped()

        # Create TwistStamped msg to publish current_velocity
        self.current_velocity_msg = TwistStamped()

        # Create a transform message
        self.t = TransformStamped()


    def transform_coordinates(self, msg):

        current_x, current_y = self.transformer.transform(msg.latitude, msg.longitude)

        pos_x = current_x - self.origin_x
        pos_y = current_y - self.origin_y

        print("Relative pose in map frame:")
        print(f"x: {pos_x}, y: {pos_y}")

        # calculate azimuth correction
        azimuth_correction = self.utm_projection.get_factors(msg.longitude, msg.latitude).meridian_convergence
        print(f"azimuth correction: {azimuth_correction}")

        azimuth_corrected = msg.azimuth + azimuth_correction
        azimuth_rad = np.deg2rad(azimuth_corrected)

        yaw = convert_azimuth_to_yaw(azimuth_rad)
        
        # Convert yaw to quaternion
        x, y, z, w = quaternion_from_euler(0, 0, yaw)
        orientation = Quaternion(x, y, z, w)

        print("Orientation:")
        print(orientation)

        # Publish current position
        self.current_pose_msg.header.stamp = msg.header.stamp
        self.current_pose_msg.header.frame_id = "map"
        self.current_pose_msg.pose.position.x = pos_x
        self.current_pose_msg.pose.position.y = pos_y
        self.current_pose_msg.pose.position.z = msg.height
        self.current_pose_msg.pose.orientation.x = x
        self.current_pose_msg.pose.orientation.y = y
        self.current_pose_msg.pose.orientation.z = z
        self.current_pose_msg.pose.orientation.w = w

        self.current_pose_pub.publish(self.current_pose_msg)

        
        # Compute current velocity and publish
        current_vel = np.sqrt((msg.north_velocity)**2 + (msg.east_velocity)**2)

        self.current_velocity_msg.header.stamp = msg.header.stamp
        self.current_velocity_msg.header.frame_id = "base_link"
        self.current_velocity_msg.twist.linear.x = current_vel

        self.current_velocity_pub.publish(self.current_velocity_msg)

        
        # Publish transform message
        self.t.header.stamp = msg.header.stamp
        self.t.header.frame_id = "map"
        self.t.child_frame_id = "base_link"
        self.t.transform.translation.x = pos_x
        self.t.transform.translation.y = pos_y
        self.t.transform.rotation.x = x
        self.t.transform.rotation.y = y
        self.t.transform.rotation.z = z
        self.t.transform.rotation.w = w

        self.br.sendTransform(self.t)

        print("------------------------------------------------------------------------------------------------")




    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('localizer')
    node = Localizer()
    node.run()