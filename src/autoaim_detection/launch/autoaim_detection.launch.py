'''
Description: 
Version: 1.0
Autor: Julian Lin
Date: 2024-04-17 21:37:14
LastEditors: Julian Lin
LastEditTime: 2024-04-18 12:46:07
'''
import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
def generate_launch_description():
    config_yaml = os.path.join(get_package_share_directory('autoaim_detection'), 'config', 'params.yaml')
    return LaunchDescription([
    Node(
        package="autoaim_detection",
        executable="autoaim_detection_node",
        output="screen",
        parameters=[config_yaml],
    )])