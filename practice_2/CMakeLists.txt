cmake_minimum_required(VERSION 3.0.2)
project(practice_2)

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
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)