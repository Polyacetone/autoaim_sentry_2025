cmake_minimum_required(VERSION 3.10)
project(autoaim_detection)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_compile_options(-Ofast -Wall -Wextra)

### <infer backend settings>
set(INFER_BACKEND_OPENVINO ON)  # use openvino
set(INFER_BACKEND_CUDA OFF) # use cuda
### </infer backend settings>

find_package(OpenMP REQUIRED)
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

ament_auto_add_library(${PROJECT_NAME} SHARED src/detection_node.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)

if(INFER_BACKEND_OPENVINO)
  add_definitions(-DINFER_BACKEND_OPENVINO)
  find_package(OpenVINO COMPONENTS Runtime)
  target_link_libraries(${PROJECT_NAME} openvino::runtime)
elseif(INFER_BACKEND_CUDA)
  # TODO
else()
  # TODO, maybe cpu?
endif()

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN autoaim_detection::YoloDetectNode
  EXECUTABLE ${PROJECT_NAME}_node
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(
  USE_SCOPED_HEADER_INSTALL_DIR
  INSTALL_TO_SHARE
  config
  launch
  model
)