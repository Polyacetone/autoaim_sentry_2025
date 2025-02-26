cmake_minimum_required(VERSION 3.8)
project(autoaim_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Detection.msg"
  "msg/DetectionArray.msg"
  "msg/ShootPos.msg"
  "msg/DecisionInfo.msg"

  "msg/ChassisStatus.msg"
  "msg/rfr/CompRobotsHp.msg"
  "msg/rfr/CompStatus.msg"
  "msg/rfr/InterMapClientToRobot.msg"
  "msg/rfr/RobotGroundPos.msg"
  "msg/rfr/RobotPerformance.msg"
  "msg/rfr/RobotPos.msg"
  "msg/rfr/RobotResource.msg"
  "msg/rfr/RobotRfid.msg"
  "msg/rfr/RobotSentryDecision.msg"
  "msg/rfr/TeamEvent.msg"
  
  DEPENDENCIES
    std_msgs
    geometry_msgs
    sensor_msgs
)

ament_package()
