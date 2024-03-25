import rospy
import math
import threading
from tf2_ros import Buffer, TransformListener, TransformException
import numpy as np
from autoware_msgs.msg import Lane, DetectedObjectArray, Waypoint
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3, Vector3Stamped
from shapely.geometry import LineString, Point, Polygon
from shapely import prepare, intersects
from tf2_geometry_msgs import do_transform_vector3
from scipy.interpolate import interp1d

class SimpleLocalPlanner:

    def __init__(self):

        # Parameters
        self.output_frame = rospy.get_param("~output_frame")
        self.local_path_length = rospy.get_param("~local_path_length")
        self.transform_timeout = rospy.get_param("~transform_timeout")
        self.braking_safety_distance_obstacle = rospy.get_param("~braking_safety_distance_obstacle")
        self.braking_safety_distance_goal = rospy.get_param("~braking_safety_distance_goal")
        self.braking_reaction_time = rospy.get_param("braking_reaction_time")
        self.stopping_lateral_distance = rospy.get_param("stopping_lateral_distance")
        self.current_pose_to_car_front = rospy.get_param("current_pose_to_car_front")
        self.default_deceleration = rospy.get_param("default_deceleration")

        # Variables
        self.lock = threading.Lock()
        self.global_path_linestring = None
        self.global_path_distances = None
        self.distance_to_velocity_interpolator = None
        self.current_speed = None
        self.current_position = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

        self.goal_point = None

        # Publishers
        self.local_path_pub = rospy.Publisher('local_path', Lane, queue_size=1, tcp_nodelay=True)

        # Subscribers
        rospy.Subscriber('global_path', Lane, self.path_callback, queue_size=None, tcp_nodelay=True)
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/localization/current_velocity', TwistStamped, self.current_velocity_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/detection/final_objects', DetectedObjectArray, self.detected_objects_callback, queue_size=1, buff_size=2**20, tcp_nodelay=True)


    def path_callback(self, msg):

        if len(msg.waypoints) == 0:
            global_path_linestring = None
            global_path_distances = None
            distance_to_velocity_interpolator = None
            rospy.loginfo("%s - Empty global path received", rospy.get_name())

        else:
            self.goal_point = Vector3Stamped()
            self.goal_point.vector = msg.waypoints[-1].pose.pose.position

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

            local_path_length = self.local_path_length
            output_frame = self.output_frame

            stopping_lateral_distance = self.stopping_lateral_distance

            default_deceleration = self.default_deceleration

            current_pose_to_car_front = self.current_pose_to_car_front
            braking_safety_distance_obstacle = self.braking_safety_distance_obstacle
            braking_reaction_time = self.braking_reaction_time

            transform_timeout = self.transform_timeout

            goal_point = self.goal_point
            braking_safety_distance_goal = self.braking_safety_distance_goal

        if global_path_linestring is None or \
           global_path_distances is None or \
           distance_to_velocity_interpolator is None or \
           current_speed is None or \
           current_position is None or \
           goal_point is None:
            
            self.publish_local_path_wp(local_path_waypoints=[],
                                       stamp=msg.header.stamp,
                                       output_frame=output_frame,
                                       )
            return
        

        d_ego_from_path_start = global_path_linestring.project(current_position)
        lane_target_velocity = distance_to_velocity_interpolator(d_ego_from_path_start)
        current_speed = min(lane_target_velocity, current_speed)

        local_path = self.extract_local_path(global_path_linestring=global_path_linestring,
                                             global_path_distances=global_path_distances,
                                             d_ego_from_path_start=d_ego_from_path_start,
                                             local_path_length=local_path_length)

        if local_path is None:
            rospy.logwarn("No local path detected")
            return

        local_path_buffer = local_path.buffer(stopping_lateral_distance, cap_style="flat")
        prepare(local_path_buffer)

        # Initializing empty lists to store the detected object polygons, distances and velocities for later use
        # to compute the target velocities
        object_polygons = []
        object_distances = []
        object_velocities = []
        
        try:
            transform = self.tf_buffer.lookup_transform(target_frame='base_link',
                                                        source_frame=output_frame,
                                                        time=msg.header.stamp,
                                                        timeout=rospy.Duration(transform_timeout))
                                                        
        except (TransformException, rospy.ROSTimeMovedBackwardsException) as e:
            rospy.logwarn("%s - %s", rospy.get_name(), e)
            transform = None
            return

        for object in msg.objects:
            object_points = [(point.x, point.y) for point in object.convex_hull.polygon.points]
            object_polygon = Polygon(object_points)
            object_polygons.append(object_polygon)

            min_dist = 100000000
            for coords in object_polygon.exterior.coords:
                d = local_path.project(Point(coords))
                if d < min_dist:
                    min_dist = d
                
            object_distances.append(min_dist)
            object_velocities.append(object.velocity.linear)
        
        # Another set of lists to store the detected obstacle distances (closest distance to base_link),
        # obstacle velocity and the braking distances to the obstacle depending on the obstacle
        # obstacle --> detected object or goal point
        obstacle_distances = []
        obstacle_velocities = []
        object_braking_distance = []

        for id, object_polygon in enumerate(object_polygons):
            if intersects(local_path_buffer, object_polygon):

                # project object velocity to base_link frame to get longitudinal speed
                # in case there is no transform assume the object is not moving
                if transform is not None:
                    vector3_stamped = Vector3Stamped(vector=object_velocities[id])
                    velocity = do_transform_vector3(vector3_stamped, transform).vector
                else:
                    velocity = Vector3()

                transformed_velocity = velocity.x

                obstacle_distances.append(object_distances[id])
                obstacle_velocities.append(transformed_velocity)

                object_braking_distance.append(braking_safety_distance_obstacle)

        # Retrieving the goal point and projecting into local path to get the distance from base_link
        goal_coord = [(goal_point.vector.x, goal_point.vector.y)]
        dist_to_goal_point = local_path.project(Point(goal_coord))

        # If the goal point is in the local path, add it as an obstacle in the lists
        # Goal point is also considered an obstacle with zero velocity, and has a braking distance of zero
        if dist_to_goal_point < local_path_length:
            obstacle_distances.append(dist_to_goal_point)
            obstacle_velocities.append(0)
            object_braking_distance.append(braking_safety_distance_goal)
        
        # converting obstacle relevant lists to numpy arrays for easy computations
        distances = np.array(obstacle_distances)
        velocities = np.array(obstacle_velocities)
        object_braking_distances = np.array(object_braking_distance)

        # Initializing the variables required for publishing local path
        target_velocity = 0.0
        closest_distance = 0.0   
        closest_object_velocity = 0.0
        stopping_point_distance = 0.0
        local_path_blocked = False

        # If there is no obstacle in local path
        # i.e., neither obstacle or goal point, then drive in the global path maintaining max target velocity permitted by lane restrictions
        # Else compute appropriate target velocities and select the appropriate velocity based on which obstacle creates the minimum one 
        if len(obstacle_distances) > 0:
            
            # computing target velocities
            target_distances = distances - current_pose_to_car_front - object_braking_distances - braking_reaction_time*np.abs(velocities)

            # Replacing all negative velocities with zero
            velocities[velocities < 0] = 0
            target_velocities_squared = velocities**2 + 2*default_deceleration*target_distances
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
            if object_braking_distances[min_target_vel_id] == braking_safety_distance_obstacle:
                local_path_blocked = True
            
            target_velocity = target_velocities[min_target_vel_id]
            closest_object_velocity = obstacle_velocities[min_target_vel_id]
            closest_distance = distances[min_target_vel_id] - current_pose_to_car_front
            stopping_point_distance = distances[min_target_vel_id] - object_braking_distances[min_target_vel_id]

        else:
            target_velocity = lane_target_velocity

        # Get the local waypoints from the local path and target velocity depending on the detected obstacles
        local_path_waypoints = self.convert_local_path_to_waypoints(local_path=local_path,
                                                            target_velocity=target_velocity)
        
        # Publish the local waypoints
        self.publish_local_path_wp(local_path_waypoints=local_path_waypoints, 
                                stamp = msg.header.stamp, 
                                output_frame = output_frame, 
                                closest_object_distance=closest_distance, 
                                closest_object_velocity=closest_object_velocity, 
                                local_path_blocked=local_path_blocked, 
                                stopping_point_distance=stopping_point_distance)


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

if __name__ == '__main__':
    rospy.init_node('simple_local_planner')
    node = SimpleLocalPlanner()
    node.run()