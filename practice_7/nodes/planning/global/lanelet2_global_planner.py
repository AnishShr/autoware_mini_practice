import rospy
import numpy as np

import lanelet2
from lanelet2.io import Origin, load
from lanelet2.projection import UtmProjector
from lanelet2.core import BasicPoint2d
from lanelet2.geometry import findNearest

from geometry_msgs.msg import PoseStamped
from autoware_msgs.msg import Lane, Waypoint

from shapely.geometry import LineString, Point

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
        self.current_location = None
        self.goal_pos = None

        # Publishers
        self.waypoints_pub = rospy.Publisher('global_path', Lane, queue_size=1, latch=True)

        # Subscribers
        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_callback, queue_size=1)
        rospy.Subscriber("/localization/current_pose", PoseStamped, self.current_pose_callback, queue_size=1)
    
    def lanelet_sequence_to_waypoints(self, lanelet_sequence):
        """
        Inputs: 
            - Lanelet sequence leading to the final waypoint in corresponding lanelet
            - User entered goal pose

        output: List of waypoints to follow without waypoints overlapping
        """
        goal_pos = self.goal_pos
        waypoints_list = []        
        
        for lanelet in lanelet_sequence:
            # Assigning speed to each lanelet, if 'speed_ref' not present
            speed = self.speed_limit
            if 'speed_ref' in lanelet.attributes:
                speed = min(float(lanelet.attributes['speed_ref']), speed)

            speed = speed / 3.6

            for point in lanelet.centerline:
                
                waypoint = Waypoint()
                waypoint.pose.pose.position.x = point.x
                waypoint.pose.pose.position.y = point.y
                waypoint.pose.pose.position.z = point.z
                waypoint.twist.twist.linear.x = speed
                
                if len(waypoints_list) == 0:
                    waypoints_list.append(waypoint)
                else:
                    if waypoint.pose != waypoints_list[-1].pose:
                        waypoints_list.append(waypoint)

        waypoints_array = np.array([(wp.pose.pose.position.x, wp.pose.pose.position.y, wp.pose.pose.position.z) for wp in waypoints_list])
        waypoints_linestring = LineString(waypoints_array[:, :2])
        
        d_goal_from_path_start = waypoints_linestring.project(goal_pos)
        goal_point_in_path = waypoints_linestring.interpolate(d_goal_from_path_start)

        waypoints_filtered = []
        for wp in waypoints_list:
            wp_pos = Point([wp.pose.pose.position.x, wp.pose.pose.position.y])
            d_wp_from_path_start = waypoints_linestring.project(wp_pos)

            if d_wp_from_path_start < d_goal_from_path_start:
                waypoints_filtered.append(wp)
            else:
                break
        
        # Getting the z coordinate of the closest waypoint to update the goal pose
        # -------------------------------------------------------------------------------------
        closest_distance = float('inf')
        closest_waypoint = None

        for wp in waypoints_array:
            x,y,_ = wp

            dist = np.sqrt((goal_point_in_path.x - x)**2 + (goal_point_in_path.y - y)**2)

            if dist < closest_distance:
                closest_distance = dist
                closest_waypoint = wp
        
        closest_z = closest_waypoint[2]
        # --------------------------------------------------------------------------------------

        goal_wp = Waypoint()
        goal_wp.pose.pose.position.x = goal_point_in_path.x
        goal_wp.pose.pose.position.y = goal_point_in_path.y
        goal_wp.pose.pose.position.z = closest_z
        waypoints_filtered.append(goal_wp)

        goal_pos = Point(goal_point_in_path.x, goal_point_in_path.y, closest_z)
        self.goal_pos = goal_pos

        return waypoints_filtered
            
    
    def create_and_publish_lane_msg(self, list_of_waypoints):

        lane = Lane()
        lane.header.frame_id = self.output_frame
        lane.header.stamp = rospy.Time.now()
        lane.waypoints = list_of_waypoints        
        
        self.waypoints_pub.publish(lane)
        


    def goal_callback(self, msg):

        if self.current_location is None:
            rospy.logwarn("%s - Current vehicle position not set !!", rospy.get_name())
            return

        rospy.loginfo("%s - goal position (%f, %f, %f) orientation (%f, %f, %f, %f) in %s frame", rospy.get_name(),
                    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                    msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z,
                    msg.pose.orientation.w, msg.header.frame_id)
        
        # Converting PoseStamped msg to BasicPoint2d geometry type
        self.goal_pos = Point(msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
        goal_point = BasicPoint2d(self.goal_pos.x, self.goal_pos.y)

        # get start and end lanelets
        start_lanelet = findNearest(self.lanelet2_map.laneletLayer, self.current_location, 1)[0][1]
        goal_lanelet = findNearest(self.lanelet2_map.laneletLayer, goal_point, 1)[0][1]
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
        # goal_pos = Point(self.goal_pos)
        waypoints = self.lanelet_sequence_to_waypoints(lanelet_sequence=path_no_lane_change)  
        # Publishing the list of waypoints in the Lane msg      
        self.create_and_publish_lane_msg(waypoints)

    
    def current_pose_callback(self, msg):

        self.current_location = BasicPoint2d(msg.pose.position.x, msg.pose.position.y)

        goal_pos = self.goal_pos
        if goal_pos is None:
            return
        
        current_pos = np.array([msg.pose.position.x, msg.pose.position.y])
        dist_ego_from_goal = np.sqrt((goal_pos.x-current_pos[0])**2 + (goal_pos.y-current_pos[1])**2)
        
        # if the ego vehicle is close to the goal, stop the vehicle
        if dist_ego_from_goal < self.distance_to_goal_limit:
            self.create_and_publish_lane_msg(list_of_waypoints=[])
            rospy.loginfo("GOAL REACHED !! \n")
            self.goal_pos = None


    def run(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('pure_pursuit_follower')
    node = LaneLet2GlobalPlanner()
    node.run()

