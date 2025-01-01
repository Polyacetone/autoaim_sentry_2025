import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
def generate_launch_description():
    config_yaml = os.path.join(get_package_share_directory('autoaim_prediction'), 'config', 'params.yaml')
    return LaunchDescription([
    Node(
        package="autoaim_prediction",
        executable="autoaim_prediction_node",
        output="screen",
        parameters=[config_yaml],
    )])