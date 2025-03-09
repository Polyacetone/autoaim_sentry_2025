import launch, time, os
from launch_ros.actions import ComposableNodeContainer, LoadComposableNodes
from launch_ros.descriptions import ComposableNode
from launch.event_handlers import (OnExecutionComplete, OnProcessExit, OnProcessIO, OnProcessStart, OnShutdown)
from launch.actions import RegisterEventHandler, LogInfo, OpaqueFunction, SetEnvironmentVariable
from ament_index_python.packages import get_package_share_directory

def launch_func(context, *args, **kwargs):
    camera_params_yaml = os.path.join(get_package_share_directory('autoaim_camera'), 'config', 'params.yaml')
    detection_params_yaml = os.path.join(get_package_share_directory('autoaim_detection'), 'config', 'params.yaml')
    prediction_params_yaml = os.path.join(get_package_share_directory('autoaim_prediction'), 'config', 'params.yaml')
    composable_node_descriptions = [
        ComposableNode(
            package='autoaim_camera',
            plugin='autoaim_camera::CameraNode',
            name='autoaim_camera',
            parameters=[camera_params_yaml],
            extra_arguments=[{'use_intra_process_comms': True}]
        ),
        ComposableNode(
            package='autoaim_detection',
            plugin='autoaim_detection::YoloDetectNode',
            name='autoaim_detection',
            parameters=[detection_params_yaml],
            extra_arguments=[{'use_intra_process_comms': True}]
        ),
        ComposableNode(
            package='autoaim_prediction',
            plugin='autoaim_prediction::PredictionNode',
            name='autoaim_prediction',
            parameters=[prediction_params_yaml],
            extra_arguments=[{'use_intra_process_comms': True}]
        )
    ]
    time.sleep(1)
    return [
        LoadComposableNodes(
            composable_node_descriptions=composable_node_descriptions, 
            target_container='autoaim_container'
        )
    ]


def generate_launch_description():
    set_env_log_format = SetEnvironmentVariable(
        name='RCUTILS_CONSOLE_OUTPUT_FORMAT',
        value='[{severity}] [{name}]: {message}'
    )
    container = ComposableNodeContainer(
        name='autoaim_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        output='both',
        emulate_tty=True,
        respawn=True,
        respawn_delay=3
    )
    event = RegisterEventHandler(
        OnProcessStart(
            target_action=container,
            on_start=[LogInfo(msg='autoaim container started'), OpaqueFunction(function=launch_func)]
        )
    )

    return launch.LaunchDescription([set_env_log_format, event, container])