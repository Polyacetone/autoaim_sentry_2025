import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
def generate_launch_description():
    config_yaml = os.path.join(get_package_share_directory('autoaim_anti_dart'), 'config', 'params.yaml')
    return LaunchDescription([
    Node(
        package="autoaim_anti_dart",
        executable="autoaim_anti_dart_node",
        namespace='',
        output="screen",
        parameters=[config_yaml],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static')
        ]
    )])