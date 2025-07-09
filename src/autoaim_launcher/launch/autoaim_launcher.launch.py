import launch, time, os
from launch_ros.actions import ComposableNodeContainer, LoadComposableNodes, PushRosNamespace, SetRemap
from launch_ros.descriptions import ComposableNode
from launch.event_handlers import (OnExecutionComplete, OnProcessExit, OnProcessIO, OnProcessStart, OnShutdown)
from launch.actions import RegisterEventHandler, LogInfo, OpaqueFunction, SetEnvironmentVariable
from ament_index_python.packages import get_package_share_directory

namespace = 'hw_sentry'
node_names = ['camera', 'detector', 'selector', 'locator', 'predictor', 'send_enemy']

def snake_to_camel(snake_str):
    parts = snake_str.split('_')
    return ''.join(word.capitalize() for word in parts)

def launch_func(context, *args, **kwargs):
    composable_node_descriptions = []
    for node_name in node_names:
        node_params_yaml = os.path.join(get_package_share_directory(f'autoaim_{node_name}'), 'config', 'params.yaml')
        composable_node_descriptions.append(
            ComposableNode(
                package=f'autoaim_{node_name}',
                plugin=f'autoaim_{node_name}::{snake_to_camel(node_name)}Node',
                name=f'autoaim_{node_name}',
                parameters=[node_params_yaml],
                extra_arguments=[{'use_intra_process_comms': True}]
            )
        )
    time.sleep(1)
    return [
        LoadComposableNodes(
            composable_node_descriptions=composable_node_descriptions, 
            target_container=(namespace, '/', 'autoaim_container')
        )
    ]

def generate_launch_description():
    set_env_log_format = SetEnvironmentVariable(
        name='RCUTILS_CONSOLE_OUTPUT_FORMAT',
        value='[{time}] [{severity}] [{name}]: {message}'
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

    return launch.LaunchDescription([
        set_env_log_format,
        PushRosNamespace(namespace=namespace),
        SetRemap('/tf', 'tf'),
        SetRemap('/tf_static', 'tf_static'),
        event,
        container
    ])