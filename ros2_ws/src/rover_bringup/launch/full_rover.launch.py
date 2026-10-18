import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource, AnyLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('rover_bringup')
    nav2_params = os.path.join(pkg_share, 'config', 'nav2_params.yaml')
    slam_params = os.path.join(pkg_share, 'config', 'mapper_params.yaml')

    # 1. Joy Node
    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        output='screen'
    )

    # 2. RPLidar Node
    rplidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('rplidar_ros'), 'launch', 'rplidar.launch.py')
        ])
    )

    # 3. Rover Master Node
    rover_master = Node(
        package='rover_bringup',
        executable='rover_master_node.py',
        name='rover_master_node',
        output='screen'
    )

    # 4. Static TF (base_link -> laser)
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '--x', '0.03',
            '--y', '0.0',
            '--z', '0.18',
            '--yaw', '0',
            '--pitch', '0',
            '--roll', '0',
            '--frame-id', 'base_link',
            '--child-frame-id', 'laser'
        ],
        output='screen'
    )

    # 5. SLAM Toolbox (Online Sync)
    slam_toolbox_launch = TimerAction(
        period=5.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([
                    os.path.join(get_package_share_directory('slam_toolbox'), 'launch', 'online_sync_launch.py')
                ]),
                launch_arguments={'slam_params_file': slam_params}.items()
            )
        ]
    )

    # 6. Nav2 Navigation Stack (delayed by 4 seconds to allow TF tree stabilization)
    nav2_launch = TimerAction(
        period=10.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([
                    os.path.join(get_package_share_directory('nav2_bringup'), 'launch', 'navigation_launch.py')
                ]),
                launch_arguments={
                    'use_sim_time': 'false',
                    'params_file': nav2_params
                }.items()
            )
        ]
    )

    # 7. Foxglove Bridge (delayed by 2 seconds)
    foxglove_launch = TimerAction(
        period=5.0,
        actions=[
            IncludeLaunchDescription(
                AnyLaunchDescriptionSource([
                    os.path.join(get_package_share_directory('foxglove_bridge'), 'launch', 'foxglove_bridge_launch.xml')
                ]),
                launch_arguments={'port': '8765'}.items()
            )
        ]
    )

    return LaunchDescription([
        joy_node,
        rplidar_launch,
        rover_master,
        static_tf,
        slam_toolbox_launch,
        foxglove_launch,
        nav2_launch
    ])

