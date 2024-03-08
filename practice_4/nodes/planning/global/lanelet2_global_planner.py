import rospy
import numpy as np

import lanelet2
from lanelet2.io import Origin, load
from lanelet2.projection import UtmProjector
from lanelet2.core import BasicPoint2d
from lanelet2.geometry import findNearest

from geometry_msgs.msg import PoseStamped
from autoware_msgs.msg import Lane, Waypoint

class LaneLet2GlobalPlanner():
    def __init__(self):
        # Parameters
        lanelet2_map_name = rospy.get_param("~lanelet2_map_name")
        
        coordinate_transformer = rospy.get_param("/localization/coordinate_transformer")
        use_custom_origin = rospy.get_param("/localization/use_custom_origin")
        utm_origin_lat = rospy.get_param("/localization/utm_origin_lat")
        utm_origin_lon = rospy.get_param("/localization/utm_origin_lon")
        
        self.speed_limit = float(rospy.get_param("~speed_limit"))        
        self.output_frame = rospy.get_param("/planning/lanelet2_global_planner/output_frame")
        self.distance_to_goal_limit = rospy.get_param("/planning/lanelet2_global_planner/distance_to_goal_limit")
        
        # Load the map using Lanelet2
        if coordinate_transformer == "utm":
            projector = UtmProjector(Origin(utm_origin_lat, utm_origin_lon), use_custom_origin, False)
        else:
            raise RuntimeError('Only "utm" is supported for lanelet2 map loading')
        self.lanelet2_map = load(lanelet2_map_name, projector)

        # traffic rules
        traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                                      lanelet2.traffic_rules.Participants.VehicleTaxi)
        # routing graph
        self.graph = lanelet2.routing.RoutingGraph(self.lanelet2_map, traffic_rules)

        # class variables
        self.goal_point = None
        self.current_location = None

        self.goal_pos = None
        self.current_pos = None

        # Publishers
        self.waypoints_pub = rospy.Publisher('/global_path', Lane, queue_size=1, latch=True)

        # Subscribers
        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_callback, queue_size=1)
        rospy.Subscriber("/localization/current_pose", PoseStamped, self.current_pose_callback, queue_size=1)
    
    def lanelet_sequence_to_waypoints(self, lanelet_sequence):
        """
        Input: Lanelet sequence leading to the final waypoint in corresponding lanelet
        output: List of waypoints to follow without waypoints overlapping
        """
        waypoint_list = []

        
        for lanelet in lanelet_sequence:
            # Assigning speed to each lanelet, if 'speed_ref' not present
            if 'speed_ref' in lanelet.attributes:
                speed = float(lanelet.attributes['speed_ref'])   # speed in km/hr
                if speed > self.speed_limit:
                    speed = self.speed_limit
                
            else:
                speed = self.speed_limit # defualt speed in km/hr
            
            # Converting speed from km/hr to m/s
            speed = speed * 1000 / (60 * 60)

            # Adding waypoints to the list, excluding overlapping of waypoints
            for point in lanelet.centerline:
                waypoint = Waypoint()
                waypoint.pose.pose.position.x = point.x
                waypoint.pose.pose.position.y = point.y
                waypoint.pose.pose.position.z = point.z
                waypoint.twist.twist.linear.x = speed

                waypoint_list.append(waypoint)
        
            waypoint_list.pop()
        waypoint_list.append(waypoint)

        return waypoint_list
            
    
    def goal_callback(self, msg):

        if self.current_location is None:
            return

        rospy.loginfo("%s - goal position (%f, %f, %f) orientation (%f, %f, %f, %f) in %s frame", rospy.get_name(),
                    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                    msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z,
                    msg.pose.orientation.w, msg.header.frame_id)
        
        # Converting PoseStamped msg to BasicPoint2d geometry type
        self.goal_pos = msg.pose
        self.goal_point = BasicPoint2d(self.goal_pos.position.x, self.goal_pos.position.y)

        # get start and end lanelets
        start_lanelet = findNearest(self.lanelet2_map.laneletLayer, self.current_location, 1)[0][1]
        goal_lanelet = findNearest(self.lanelet2_map.laneletLayer, self.goal_point, 1)[0][1]
        # find routing graph
        route = self.graph.getRoute(start_lanelet, goal_lanelet, 0, True)

        if route is None:
            rospy.logwarn("No Route Found !!")
            return

        # find shortest path
        path = route.shortestPath()
        # this returns LaneletSequence to a point where lane change would be necessary to continue
        path_no_lane_change = path.getRemainingLane(start_lanelet)

        # Getting the list of waypoints
        waypoints = self.lanelet_sequence_to_waypoints(path_no_lane_change)
        
        # Replace the final point in the waypoints list by the goal point
        waypoints[-1].pose.pose = self.goal_pos

        # Publish the lane msg with waypoints acquired
        lane = Lane()
        lane.header.frame_id = self.output_frame
        lane.header.stamp = rospy.Time.now()
        lane.waypoints = waypoints
        self.waypoints_pub.publish(lane)
    
    def current_pose_callback(self, msg):

        self.current_location = BasicPoint2d(msg.pose.position.x, msg.pose.position.y)

        if self.goal_pos is None:
            return

        current_pos = np.array([msg.pose.position.x, msg.pose.position.y])
        goal_pos = np.array([self.goal_pos.position.x, self.goal_pos.position.y])

        # Compute the distance of the vehicle from the goal
        dist_ego_from_goal = np.sqrt((goal_pos[0]-current_pos[0])**2 + (goal_pos[1]-current_pos[1])**2)

        # Clear the path(waypoints) once the vehicle is within distance limit to the goal 
        if dist_ego_from_goal < self.distance_to_goal_limit:
            lane = Lane()
            lane.header.frame_id = self.output_frame
            lane.header.stamp = rospy.Time.now()
            self.waypoints_pub.publish(lane)

            rospy.loginfo("GOAL REACHED !!")
            self.goal_pos = None
            return


    def run(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('pure_pursuit_follower')
    node = LaneLet2GlobalPlanner()
    node.run()

