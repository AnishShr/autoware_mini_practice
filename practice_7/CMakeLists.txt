cmake_minimum_required(VERSION 3.0.2)
project(practice_7)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
)

catkin_package(
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  nodes/localization/localizer.py
  nodes/control/pure_pursuit_follower.py  
  nodes/detection/lidar_cluster/points_clusterer.py
  nodes/detection/lidar_cluster/cluster_detector.py
  nodes/detection/tfl_camera/camera_traffic_light_detector.py
  nodes/planning/global/lanelet2_global_planner.py
  nodes/planning/local/simple_local_planner.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)