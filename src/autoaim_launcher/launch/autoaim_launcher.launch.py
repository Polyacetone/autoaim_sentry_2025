import os
from ament_index_python.packages import get_package_share_directory, get_package_prefix
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    camera_params_yaml = os.path.join(get_package_share_directory('autoaim_camera'), 'config', 'params.yaml')
    detection_params_yaml = os.path.join(get_package_share_directory('autoaim_detection'), 'config', 'params.yaml')
    prediction_params_yaml = os.path.join(get_package_share_directory('autoaim_prediction'), 'config', 'params.yaml')
    tf_remappings = [
        ('/tf', 'tf'),
        ('/tf_static', 'tf_static')
    ]
    return LaunchDescription([
        Node(
            package='autoaim_camera',
            executable='autoaim_camera_node',
            name='autoaim_camera',
            namespace='hw_sentry',
            output='screen',
            emulate_tty=True,
            parameters=[camera_params_yaml],
            remappings=tf_remappings,
        ),
        Node(
            package="autoaim_detection",
            executable="autoaim_detection_node",
            name='autoaim_detection',
            namespace='hw_sentry',
            output="screen",
            emulate_tty=True,
            parameters=[detection_params_yaml],
            remappings=tf_remappings,
        ),
        Node(
            package='autoaim_prediction',
            executable='autoaim_prediction_node',
            name='autoaim_prediction',
            namespace='hw_sentry',
            output='screen',
            emulate_tty=True,
            parameters=[prediction_params_yaml],
            remappings=tf_remappings,
        ),
    ])