cmake_minimum_required(VERSION 3.10)
project(autoaim_prediction)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_compile_options(-Ofast -fno-unsafe-math-optimizations)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()
include_directories(/usr/include/eigen3)

ament_auto_add_library(${PROJECT_NAME} SHARED src/prediction_node.cpp)

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN autoaim_prediction::PredictionNode
  EXECUTABLE ${PROJECT_NAME}_node
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(
  INSTALL_TO_SHARE
  config
  launch
)