import rospy
import numpy as np

from tf.transformations import euler_from_quaternion
from autoware_msgs.msg import Lane
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from autoware_msgs.msg import VehicleCmd

from shapely.geometry import LineString, Point
from shapely import prepare, distance
from scipy.interpolate import interp1d



class PurePursuitFollower:
    def __init__(self):

        # Parameters
        self.wheel_base = float(rospy.get_param('/wheel_base'))        
        self.lookahead_distance = float(rospy.get_param('/pure_pursuit_follower/lookahead_distance'))

        # class variables
        self.init_pose = None
        self.path_linestring = None
        self.distance_to_velocity_interpolator = None

        # Publishers
        self.current_velocity_pub = rospy.Publisher('control/vehicle_cmd', VehicleCmd, queue_size=1)

        # Subscribers
        rospy.Subscriber('path', Lane, self.path_callback, queue_size=1)
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback, queue_size=1)
        rospy.Subscriber('/initialpose', PoseWithCovarianceStamped, self.initialpose_callback, queue_size=None, tcp_nodelay=True)
     

    def initialpose_callback(self, msg):
        self.init_pose = msg


    def path_callback(self, msg):
        # TODO
        # convert waypoints to shapely linestring
        path_linestring = LineString([(w.pose.pose.position.x, w.pose.pose.position.y) for w in msg.waypoints])
        # prepare path - creates spatial tree, making the spatial queries more efficient
        prepare(path_linestring)
        self.path_linestring = path_linestring

        # collect waypoint x and y coordinates
        waypoints_xy = np.array([(w.pose.pose.position.x, w.pose.pose.position.y) for w in msg.waypoints])

        # Calculate distances between points
        distances = np.cumsum(np.sqrt(np.sum(np.diff(waypoints_xy, axis=0)**2, axis=1)))
        
        # add 0 distance in the beginning
        distances = np.insert(distances, 0, 0)
        
        # Extract velocity values at waypoints
        velocities = np.array([w.twist.twist.linear.x for w in msg.waypoints])

        # distance to velocity interpolator
        self.distance_to_velocity_interpolator = interp1d(distances, velocities, kind='linear')


    def current_pose_callback(self, msg):

        # Publish velcoities only when the initial pose is set and the distance to velocity interpolator is assigned a value
        if self.distance_to_velocity_interpolator is not None and self.init_pose is not None:

            current_pose = Point([msg.pose.position.x, msg.pose.position.y])
            d_ego_from_path_start = self.path_linestring.project(current_pose)

            # using euler_from_quaternion to get the heading angle
            _, _, heading = euler_from_quaternion([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])

            # dist from start of path to the lookahead distance
            d_lookahead_point = d_ego_from_path_start + self.lookahead_distance

            lookahead_point = self.path_linestring.interpolate(d_lookahead_point)
            ld = distance(current_pose, lookahead_point)

            # lookahead point heading calculation
            lookahead_heading = np.arctan2(lookahead_point.y - current_pose.y, lookahead_point.x - current_pose.x)
            
            # Compute the linear velocity and steering angle, and publish    
            velocity = self.distance_to_velocity_interpolator(d_ego_from_path_start)
            
            alpha = lookahead_heading - heading
            steering_angle = np.arctan2((2*self.wheel_base)*np.sin(alpha), ld)

            vehicle_cmd = VehicleCmd()
            vehicle_cmd.header.stamp = msg.header.stamp
            vehicle_cmd.header.frame_id = "base_link"
            vehicle_cmd.ctrl_cmd.steering_angle = steering_angle
            vehicle_cmd.ctrl_cmd.linear_velocity = velocity

            self.current_velocity_pub.publish(vehicle_cmd)       
  

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('pure_pursuit_follower')
    node = PurePursuitFollower()
    node.run()