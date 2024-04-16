import rospy
import math
import threading
from tf2_ros import Buffer, TransformListener, TransformException
import numpy as np
from autoware_msgs.msg import Lane, DetectedObjectArray, Waypoint
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3, Vector3Stamped
from autoware_msgs.msg import TrafficLightResultArray
from lanelet2.io import Origin, load
from lanelet2.projection import UtmProjector
from shapely.geometry import LineString, Point, Polygon
from shapely import prepare, intersects, intersection
from tf2_geometry_msgs import do_transform_vector3
from scipy.interpolate import interp1d
from numpy.lib.recfunctions import unstructured_to_structured

class SimpleLocalPlanner:

    def __init__(self):

        # Parameters
        self.output_frame = rospy.get_param("~output_frame")
        self.local_path_length = rospy.get_param("~local_path_length")
        self.transform_timeout = rospy.get_param("~transform_timeout")
        self.braking_safety_distance_obstacle = rospy.get_param("~braking_safety_distance_obstacle")
        self.braking_safety_distance_goal = rospy.get_param("~braking_safety_distance_goal")
        self.braking_safety_distance_stopline = rospy.get_param("~braking_safety_distance_stopline")
        self.braking_reaction_time = rospy.get_param("braking_reaction_time")
        self.stopping_lateral_distance = rospy.get_param("stopping_lateral_distance")
        self.current_pose_to_car_front = rospy.get_param("current_pose_to_car_front")
        self.tfl_maximum_deceleration = rospy.get_param("~tfl_maximum_deceleration")
        self.default_deceleration = rospy.get_param("default_deceleration")
        
        # Parameters related to lanelet2 map loading
        coordinate_transformer = rospy.get_param("/localization/coordinate_transformer")
        use_custom_origin = rospy.get_param("/localization/use_custom_origin")
        utm_origin_lat = rospy.get_param("/localization/utm_origin_lat")
        utm_origin_lon = rospy.get_param("/localization/utm_origin_lon")
        lanelet2_map_name = rospy.get_param("~lanelet2_map_name")

        # Variables
        self.lock = threading.Lock()
        self.global_path_linestring = None
        self.global_path_distances = None
        self.distance_to_velocity_interpolator = None
        self.current_speed = None
        self.current_position = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)
        self.red_tfl_id = None

        # Load the map using Lanelet2
        if coordinate_transformer == "utm":
            projector = UtmProjector(Origin(utm_origin_lat, utm_origin_lon), use_custom_origin, False)
        else:
            raise RuntimeError('Only "utm" is supported for lanelet2 map loading')
        lanelet2_map = load(lanelet2_map_name, projector)

        # Extract all stop lines and signals from the lanelet2 map
        all_stoplines = get_stoplines(lanelet2_map)
        self.signals = get_stoplines_trafficlights_bulbs(lanelet2_map)
        # If stopline_id is not in self.signals then it has no signals (traffic lights)
        self.tfl_stoplines = {k: v for k, v in all_stoplines.items() if k in self.signals}

        # Publishers
        self.local_path_pub = rospy.Publisher('local_path', Lane, queue_size=1, tcp_nodelay=True)

        # Subscribers
        rospy.Subscriber('global_path', Lane, self.path_callback, queue_size=None, tcp_nodelay=True)
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/localization/current_velocity', TwistStamped, self.current_velocity_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/detection/final_objects', DetectedObjectArray, self.detected_objects_callback, queue_size=1, buff_size=2**20, tcp_nodelay=True)
        rospy.Subscriber('/detection/traffic_light_status', TrafficLightResultArray, self.tfl_status_callback, queue_size=1, tcp_nodelay=True)


    def path_callback(self, msg):

        if len(msg.waypoints) == 0:
            global_path_linestring = None
            global_path_distances = None
            distance_to_velocity_interpolator = None
            rospy.loginfo("%s - Empty global path received", rospy.get_name())

        else:
            waypoints_xyz = np.array([(w.pose.pose.position.x, w.pose.pose.position.y, w.pose.pose.position.z) for w in msg.waypoints])
            # convert waypoints to shapely linestring
            global_path_linestring = LineString(waypoints_xyz)
            prepare(global_path_linestring)

            # calculate distances between points, use only xy, and insert 0 at start of distances array
            global_path_distances = np.cumsum(np.sqrt(np.sum(np.diff(waypoints_xyz[:,:2], axis=0)**2, axis=1)))
            global_path_distances = np.insert(global_path_distances, 0, 0)

            # extract velocity values at waypoints
            velocities = np.array([w.twist.twist.linear.x for w in msg.waypoints])
            # create interpolator
            distance_to_velocity_interpolator = interp1d(global_path_distances, velocities, kind='linear', bounds_error=False, fill_value=0.0)

            rospy.loginfo("%s - Global path received with %i waypoints", rospy.get_name(), len(msg.waypoints))

        with self.lock:
            self.global_path_linestring = global_path_linestring
            self.global_path_distances = global_path_distances
            self.distance_to_velocity_interpolator = distance_to_velocity_interpolator

    def current_velocity_callback(self, msg):
        # save current velocity
        self.current_speed = msg.twist.linear.x

    def current_pose_callback(self, msg):
        # save current pose
        current_position = Point([msg.pose.position.x, msg.pose.position.y])
        self.current_position = current_position

    def detected_objects_callback(self, msg):

        with self.lock:
            global_path_linestring = self.global_path_linestring
            global_path_distances = self.global_path_distances
            distance_to_velocity_interpolator = self.distance_to_velocity_interpolator
            current_speed = self.current_speed
            current_position = self.current_position

        if global_path_linestring is None or global_path_distances is None or distance_to_velocity_interpolator is None \
                or current_speed is None or current_position is None :
            
            self.publish_local_path_wp(local_path_waypoints=[],
                                       stamp=msg.header.stamp,
                                       output_frame=self.output_frame,
                                       )
            return
        

        d_ego_from_path_start = global_path_linestring.project(current_position)
        lane_target_velocity = distance_to_velocity_interpolator(d_ego_from_path_start)

        local_path = self.extract_local_path(global_path_linestring=global_path_linestring,
                                             global_path_distances=global_path_distances,
                                             d_ego_from_path_start=d_ego_from_path_start,
                                             local_path_length=self.local_path_length)

        if local_path is None:
            rospy.logwarn("%s - No local path detected", rospy.get_name())
            self.publish_local_path_wp(local_path_waypoints=[],
                                       stamp=msg.header.stamp,
                                       output_frame=self.output_frame,
                                       )
            return

        local_path_buffer = local_path.buffer(self.stopping_lateral_distance, cap_style="flat")
        prepare(local_path_buffer)

        # Initializing empty lists to store the detected object's distances and velocities for later use
        # to compute the target velocities
        object_distances = []
        object_velocities = []
        object_braking_distances = []
        
        try:
            transform = self.tf_buffer.lookup_transform(target_frame='base_link',
                                                        source_frame=self.output_frame,
                                                        time=msg.header.stamp,
                                                        timeout=rospy.Duration(self.transform_timeout))
                                                        
        except (TransformException, rospy.ROSTimeMovedBackwardsException) as e:
            rospy.logwarn("%s - %s", rospy.get_name(), e)
            transform = None
            

        for object in msg.objects:
            object_points = [(point.x, point.y) for point in object.convex_hull.polygon.points]
            object_polygon = Polygon(object_points)
            
            min_dist = float('inf')
            d = float('inf')

            for coords in object_polygon.exterior.coords:
                if intersects(local_path_buffer, Point(coords)):
                    d = local_path.project(Point(coords))
                    d = min(min_dist, d)
            
            if d <= self.local_path_length:

                object_velocity = object.velocity.linear

                if transform is not None:
                    vector3_stamped = Vector3Stamped(vector=object_velocity)
                    velocity = do_transform_vector3(vector3_stamped, transform).vector
                else:
                    velocity = Vector3()
                
                transformed_velocity = velocity.x
                
                object_distances.append(d)
                object_velocities.append(transformed_velocity)
                object_braking_distances.append(self.braking_safety_distance_obstacle)

        if self.red_tfl_id is not None:
            red_tfl_id = self.red_tfl_id
            red_tfl_line = self.tfl_stoplines[red_tfl_id]

            if intersects(local_path_buffer, red_tfl_line):
                intersection_geometry = local_path_buffer.intersection(red_tfl_line)
                intersection_point = intersection_geometry.centroid

                d_red_tfl_line = local_path.project(intersection_point)
                vel_red_tfl_line = 0        # obstacle velocity == 0 for stoplines

                current_deceleration = (current_speed**2)/(2*(d_red_tfl_line - self.braking_safety_distance_stopline))        

                if current_deceleration < self.braking_safety_distance_stopline:
                    object_distances.append(d_red_tfl_line)
                    object_velocities.append(vel_red_tfl_line)
                    object_braking_distances.append(self.braking_safety_distance_stopline)
                else:
                    rospy.logwarn_throttle(3.0, f"{rospy.get_name()} - ignore red traffic lights, deceleration: {current_deceleration}")

        # distance from ego vehicle to the goal point
        dist_to_goal_point = global_path_distances[-1] - d_ego_from_path_start

        # If the goal point is in the local path, add it as an obstacle in the lists
        # Goal point is also considered an obstacle with zero velocity, and has a braking distance of zero
        if dist_to_goal_point <= self.local_path_length:
            object_distances.append(dist_to_goal_point)
            object_velocities.append(0)
            object_braking_distances.append(self.braking_safety_distance_goal)
        
        # converting obstacle relevant lists to numpy arrays for easy computations
        object_distances = np.array(object_distances)
        object_velocities = np.array(object_velocities)
        object_braking_distances = np.array(object_braking_distances)

        # Initializing the variables required for publishing local path
        target_velocity = lane_target_velocity
        closest_object_distance = 0.0   
        closest_object_velocity = 0.0
        stopping_point_distance = 0.0
        local_path_blocked = False

        # If there is no obstacle in local path
        # i.e., neither obstacle or goal point, then drive in the global path maintaining max target velocity permitted by lane restrictions
        # Else compute appropriate target velocities and select the appropriate velocity based on which obstacle creates the minimum one 
        if len(object_distances) > 0:
            
            # computing target velocities
            target_distances = object_distances - self.current_pose_to_car_front - object_braking_distances - self.braking_reaction_time*np.abs(object_velocities)

            # Replacing all negative velocities with zero
            object_velocities[object_velocities < 0] = 0
            target_velocities_squared = object_velocities**2 + 2*self.default_deceleration*target_distances
            
            # Making sure there is no negative values inside the square root
            target_velocities_squared[target_velocities_squared < 0] = 0
            target_velocities = np.sqrt(target_velocities_squared)

            # If the target velocity exceeds the target velocity permitted by lane restrictions,
            # limit the target velocity to the lane's max permitted velocity
            target_velocities[target_velocities > lane_target_velocity] = lane_target_velocity
            
            # Getting the index of the obstacle that creates min target velocity
            min_target_vel_id = np.argmin(target_velocities)
            
            # If the obstacle that creates min target velocity is not the goal point, then set local_path_blacked to True
            # Else it should be False
            if object_braking_distances[min_target_vel_id] == self.braking_safety_distance_obstacle:
                local_path_blocked=True

            target_velocity = target_velocities[min_target_vel_id]
            closest_object_velocity = object_velocities[min_target_vel_id]
            closest_object_distance = object_distances[min_target_vel_id]
            stopping_point_distance = object_distances[min_target_vel_id] - object_braking_distances[min_target_vel_id]

        # Get the local waypoints from the local path and target velocity depending on the detected obstacles
        local_path_waypoints = self.convert_local_path_to_waypoints(local_path=local_path,
                                                            target_velocity=target_velocity)

        # Publish the local waypoints
        self.publish_local_path_wp(local_path_waypoints=local_path_waypoints, 
                                stamp = msg.header.stamp, 
                                output_frame = self.output_frame, 
                                closest_object_distance=closest_object_distance, 
                                closest_object_velocity=closest_object_velocity, 
                                local_path_blocked=local_path_blocked,
                                stopping_point_distance=stopping_point_distance)


    def tfl_status_callback(self, msg):
        # print(msg)

        red_tfl_id = None

        for result in msg.results:
            recognition_result = result.recognition_result
            if recognition_result == 0:
                red_tfl_id = result.lane_id
        
        self.red_tfl_id = red_tfl_id


    def extract_local_path(self, global_path_linestring, global_path_distances, d_ego_from_path_start, local_path_length):

        # current position is projected at the end of the global path - goal reached
        if math.isclose(d_ego_from_path_start, global_path_linestring.length):
            return None

        d_to_local_path_end = d_ego_from_path_start + local_path_length

        # find index where distances are higher than ego_d_on_global_path
        index_start = np.argmax(global_path_distances >= d_ego_from_path_start)
        index_end = np.argmax(global_path_distances >= d_to_local_path_end)

        # if end point of local_path is past the end of the global path (returns 0) then take index of last point
        if index_end == 0:
            index_end = len(global_path_linestring.coords) - 1

        # create local path from global path add interpolated points at start and end, use sliced point coordinates in between
        start_point = global_path_linestring.interpolate(d_ego_from_path_start)
        end_point = global_path_linestring.interpolate(d_to_local_path_end)
        local_path = LineString([start_point] + list(global_path_linestring.coords[index_start:index_end]) + [end_point])

        return local_path


    def convert_local_path_to_waypoints(self, local_path, target_velocity):
        # convert local path to waypoints
        local_path_waypoints = []
        for point in local_path.coords:
            waypoint = Waypoint()
            waypoint.pose.pose.position.x = point[0]
            waypoint.pose.pose.position.y = point[1]
            waypoint.pose.pose.position.z = point[2]
            waypoint.twist.twist.linear.x = target_velocity
            local_path_waypoints.append(waypoint)
        return local_path_waypoints


    def publish_local_path_wp(self, local_path_waypoints, stamp, output_frame, closest_object_distance=0.0, closest_object_velocity=0.0, local_path_blocked=False, stopping_point_distance=0.0):
        # create lane message
        lane = Lane()
        lane.header.frame_id = output_frame
        lane.header.stamp = stamp
        lane.waypoints = local_path_waypoints
        lane.closest_object_distance = closest_object_distance
        lane.closest_object_velocity = closest_object_velocity
        lane.is_blocked = local_path_blocked
        lane.cost = stopping_point_distance
        self.local_path_pub.publish(lane)


    def run(self):
        rospy.spin()


def get_stoplines(lanelet2_map):
    """
    Add all stop lines to a dictionary with stop_line id as key and stop_line as value
    :param lanelet2_map: lanelet2 map
    :return: {stop_line_id: stopline, ...}
    """

    stoplines = {}
    for line in lanelet2_map.lineStringLayer:
        if line.attributes:
            if line.attributes["type"] == "stop_line":
                # add stoline to dictionary and convert it to shapely LineString
                stoplines[line.id] = LineString([(p.x, p.y) for p in line])

    return stoplines

def get_stoplines_trafficlights_bulbs(lanelet2_map):
    """
    Iterate over all regulatory_elements with subtype traffic light and extract the stoplines and sinals.
    Organize the data into dictionary indexed by stopline id that contains a traffic_light id and light bulb data.
    :param lanelet2_map: lanelet2 map
    :return: {stopline_id: {traffic_light_id: [[bulb_id, bulb_color, x, y, z], ...], ...}, ...}
    """

    signals = {}

    for reg_el in lanelet2_map.regulatoryElementLayer:
        if reg_el.attributes["subtype"] == "traffic_light":
            # ref_line is the stop line and there is only 1 stopline per traffic light reg_el
            linkId = reg_el.parameters["ref_line"][0].id

            for bulbs in reg_el.parameters["light_bulbs"]:
                # plId represents the traffic light (pole), one stop line can be associated with multiple traffic lights
                plId = bulbs.id
                # one traffic light has red, yellow and green bulbs
                bulb_data = [[bulb.id, bulb.attributes["color"], bulb.x, bulb.y, bulb.z] for bulb in bulbs]
                # signals is a dictionary indexed by stopline id and contains dictionary of traffic lights indexed by pole id
                # which in turn contains a list of bulbs
                signals.setdefault(linkId, {}).setdefault(plId, []).extend(bulb_data)

    return signals


if __name__ == '__main__':
    rospy.init_node('simple_local_planner')
    node = SimpleLocalPlanner()
    node.run()