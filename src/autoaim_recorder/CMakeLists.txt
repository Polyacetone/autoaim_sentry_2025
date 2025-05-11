cmake_minimum_required(VERSION 3.10)
project(autoaim_recorder)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_compile_options(-Ofast -Wall -Wextra)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

ament_auto_add_library(${PROJECT_NAME} SHARED src/recorder_node.cpp)

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN autoaim_recorder::RecorderNode
  EXECUTABLE ${PROJECT_NAME}_node
)

ament_auto_package(
  USE_SCOPED_HEADER_INSTALL_DIR
  INSTALL_TO_SHARE
  config
  launch
)