cmake_minimum_required(VERSION 3.8)
project(rover_bringup)

find_package(ament_cmake REQUIRED)

# Install Python node into lib/rover_bringup
install(PROGRAMS
  scripts/rover_master_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch and config directories into share/rover_bringup
install(DIRECTORY
  config
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

