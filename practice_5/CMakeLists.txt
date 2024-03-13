cmake_minimum_required(VERSION 3.0.2)
project(practice_5)

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
  nodes/planning/global/lanelet2_global_planner.py
  nodes/detection/lidar_cluster/points_clusterer.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
