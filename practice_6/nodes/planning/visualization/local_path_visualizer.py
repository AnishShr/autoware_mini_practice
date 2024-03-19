#!/usr/bin/env python3

import rospy
import math
from autoware_msgs.msg import Lane
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point, Quaternion

from tf.transformations import quaternion_from_euler

class LocalPathVisualizer:
    def __init__(self):

        # Parameters
        self.stopping_lateral_distance = rospy.get_param("stopping_lateral_distance")
        self.current_pose_to_car_front = rospy.get_param("current_pose_to_car_front")
        self.stopping_speed_limit = rospy.get_param("stopping_speed_limit")

        # Publishers
        self.local_path_markers_pub = rospy.Publisher('local_path_markers', MarkerArray, queue_size=1, tcp_nodelay=True)

        # Subscribers
        rospy.Subscriber('local_path', Lane, self.local_path_callback, queue_size=1, buff_size=2**20, tcp_nodelay=True)


    def local_path_callback(self, lane):

        # lane.cost is used to determine the stopping point distance from path start
        stopping_point_distance = lane.cost

        marker_array = MarkerArray()

        # create marker_array to delete previous visualization
        marker = Marker()
        marker.header.frame_id = lane.header.frame_id
        marker.action = Marker.DELETEALL
        marker_array.markers.append(marker)

        stamp = rospy.Time.now()

        if len(lane.waypoints) > 1:

            points = [waypoint.pose.pose.position for waypoint in lane.waypoints]

            # Color of the local path
            color = ColorRGBA(0.2, 1.0, 0.2, 0.3)

            # local path with stopping_lateral_distance
            marker = Marker()
            marker.header.frame_id = lane.header.frame_id
            marker.header.stamp = stamp
            marker.ns = "Stopping lateral distance"
            marker.type = marker.LINE_STRIP
            marker.action = marker.ADD
            marker.id = 0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 2*self.stopping_lateral_distance
            marker.color = color
            marker.points = points
            marker_array.markers.append(marker)


            # velocity labels
            for i, waypoint in enumerate(lane.waypoints):
                marker = Marker()
                marker.header.frame_id = lane.header.frame_id
                marker.header.stamp = stamp
                marker.ns = "Velocity label"
                marker.id = i
                marker.type = marker.TEXT_VIEW_FACING
                marker.action = marker.ADD
                marker.pose = waypoint.pose.pose
                marker.scale.z = 0.5
                marker.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)
                marker.text = str(round(waypoint.twist.twist.linear.x * 3.6, 1))
                marker_array.markers.append(marker)
                # add only up to a first 0.0 velocity label
                if math.isclose(waypoint.twist.twist.linear.x, 0.0):
                    break

            if stopping_point_distance > 0.0:
                stop_position, stop_orientation = get_point_and_orientation_on_path_within_distance(lane.waypoints, 1, lane.waypoints[0].pose.pose.position, stopping_point_distance)

                color = ColorRGBA(0.9, 0.9, 0.9, 0.2)           # white - obstcle affecting ego speed in slowdown area
                if lane.is_blocked:
                    color = ColorRGBA(1.0, 1.0, 0.0, 0.5)       # yellow - obstacle in stopping area
                    if lane.closest_object_velocity < self.stopping_speed_limit:
                        color = ColorRGBA(1.0, 0.0, 0.0, 0.5)   # red - obstacle in front and very slow

                # "Stopping point" - obstacle that currently causes the smallest target velocity
                marker = Marker()
                marker.header.frame_id = lane.header.frame_id
                marker.header.stamp = stamp
                marker.ns = "Stopping point"
                marker.id = 0
                marker.type = marker.CUBE
                marker.action = marker.ADD
                marker.pose.position.x = stop_position.x
                marker.pose.position.y = stop_position.y
                marker.pose.position.z = stop_position.z + 1.0
                marker.pose.orientation = stop_orientation
                marker.scale.x = 0.3
                marker.scale.y = 5.0
                marker.scale.z = 2.5
                marker.color = color
                marker_array.markers.append(marker)

        self.local_path_markers_pub.publish(marker_array)


    def run(self):
        rospy.spin()


def get_point_and_orientation_on_path_within_distance(waypoints, front_wp_idx, start_point, distance):
    """
    Get point on path within distance from ego pose
    :param waypoints: waypoints
    :param front_wp_idx: wp index from where to start calculate the distance
    :param start_point: starting point for distance calculation
    :param distance: distance where to find the point on the path
    :return: Point, Quaternion
    """

    point = Point()
    last_idx = len(waypoints) - 1

    i = front_wp_idx
    d = get_distance_between_two_points_2d(start_point, waypoints[i].pose.pose.position)
    while d < distance:
        i += 1
        d += get_distance_between_two_points_2d(waypoints[i-1].pose.pose.position, waypoints[i].pose.pose.position)
        if i == last_idx:
            break

    # Find point orientation and distance difference and correct along path backwards
    end_orientation = math.atan2(waypoints[i - 1].pose.pose.position.y - waypoints[i].pose.pose.position.y, waypoints[i - 1].pose.pose.position.x - waypoints[i].pose.pose.position.x)
    dx = (distance - d) * math.cos(end_orientation)
    dy = (distance - d) * math.sin(end_orientation)
    point.x = waypoints[i].pose.pose.position.x - dx
    point.y = waypoints[i].pose.pose.position.y - dy
    point.z = waypoints[i].pose.pose.position.z

    x, y, z, w = quaternion_from_euler(0, 0, end_orientation)
    orientation = Quaternion(x, y, z, w)

    return point, orientation


def get_distance_between_two_points_2d(p1, p2):
    """
    Get distance between two points
    :param point1: Point
    :param point2: Point
    :return: distance
    """

    return math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)


if __name__ == '__main__':
    rospy.init_node('local_path_visualizer', log_level=rospy.INFO)
    node = LocalPathVisualizer()
    node.run()