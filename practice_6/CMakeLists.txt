cmake_minimum_required(VERSION 2.8.12...3.25.2)
project(practice_6)

find_package(catkin REQUIRED)
catkin_package()

catkin_install_python(PROGRAMS
  nodes/localization/localizer.py
  nodes/control/pure_pursuit_follower.py  
  nodes/detection/lidar_cluster/points_clusterer.py
  nodes/detection/lidar_cluster/cluster_detector.py
  nodes/planning/global/lanelet2_global_planner.py
  nodes/planning/local/simple_local_planner.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

