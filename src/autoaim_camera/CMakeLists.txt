cmake_minimum_required(VERSION 3.8)
project(autoaim_camera)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_compile_options("-Ofast")

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

ament_auto_add_library(${PROJECT_NAME} SHARED
  src/camera_node.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC hikSDK/include)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  target_link_directories(${PROJECT_NAME} PUBLIC hikSDK/lib/arm64)
  install(
    DIRECTORY hikSDK/lib/arm64/
    DESTINATION lib
  )
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64" OR "AMD64")
  target_link_directories(${PROJECT_NAME} PUBLIC hikSDK/lib/x64)
  install(
    DIRECTORY hikSDK/lib/x64/
    DESTINATION lib
  )
endif()

target_link_libraries(${PROJECT_NAME}
  FormatConversion
  MediaProcess
  MvCameraControl
  MVRender
  MvUsb3vTL
)

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN autoaim_camera::CameraNode
  EXECUTABLE ${PROJECT_NAME}_node
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  list(APPEND AMENT_LINT_AUTO_EXCLUDE
    ament_cmake_copyright
    ament_cmake_cpplint
    ament_cmake_uncrustify
  )
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(
  INSTALL_TO_SHARE
  config
  launch
)
