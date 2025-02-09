import os
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch.substitutions import LaunchConfiguration
from launch.actions import Shutdown, SetEnvironmentVariable
from ament_index_python.packages import get_package_share_directory

use_intra_process_comms = True

def generate_launch_description():
    camera_params_yaml = os.path.join(get_package_share_directory('autoaim_camera'), 'config', 'camera_params.yaml')
    detection_params_yaml = os.path.join(get_package_share_directory('autoaim_detection'), 'config', 'params.yaml')
    prediction_params_yaml = os.path.join(get_package_share_directory('autoaim_prediction'), 'config', 'params.yaml')
    serial_params_yaml = os.path.join(get_package_share_directory('autoaim_serial_driver'), 'config', 'params.yaml')

    set_env_log_format = SetEnvironmentVariable(
        name='RCUTILS_CONSOLE_OUTPUT_FORMAT',
        value='[{severity}] [{name}]: {message}'
    )
    
    container = ComposableNodeContainer(
        name='autoaim',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            ComposableNode(
                package='autoaim_camera',
                plugin='autoaim_camera::CameraNode',
                name='autoaim_camera',
                parameters=[camera_params_yaml],
                extra_arguments=[{'use_intra_process_comms': use_intra_process_comms}]
            ),
            ComposableNode(
                package='autoaim_detection',
                plugin='autoaim_detection::YoloDetectNode',
                name='autoaim_detection',
                parameters=[detection_params_yaml],
                extra_arguments=[{'use_intra_process_comms': use_intra_process_comms}]
            ),
            ComposableNode(
                package='autoaim_prediction',
                plugin='autoaim_prediction::PredictionNode',
                name='autoaim_prediction',
                parameters=[prediction_params_yaml],
                extra_arguments=[{'use_intra_process_comms': use_intra_process_comms}]
            ),
            ComposableNode(
                package='autoaim_serial_driver',
                plugin='autoaim_serial_driver::SerialDriverNode',
                name='autoaim_serial_driver',
                parameters=[serial_params_yaml],
                extra_arguments=[{'use_intra_process_comms': use_intra_process_comms}]
            )
        ],
        output='both',
        emulate_tty=True,
        on_exit=Shutdown(),
    )
    
    return LaunchDescription([set_env_log_format, container])