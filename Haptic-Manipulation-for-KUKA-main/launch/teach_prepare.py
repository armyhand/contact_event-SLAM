# please conncet to KUKA first
#
# ros2 launch lbr_bringup hardware.launch.py \
# ctrl:=lbr_joint_position_command_controller \
# model:=iiwa14
from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

# 需要 reset
# ros2 launch haptic teach_prepare.launch.py reset:=true
# 不需要 reset
# ros2 launch haptic teach_prepare.launch.py reset:=false


def generate_launch_description():
    pkg_lbr = get_package_share_directory('lbr_demos_advanced_py')
    admittance_param = os.path.join(pkg_lbr, 'config', 'admittance_control.yaml')

    do_reset = LaunchConfiguration('reset')

    return LaunchDescription([
        DeclareLaunchArgument(
            'reset',
            default_value='true',
            description='whether to run haptic reset'
        ),

        # 阻抗控制
        ExecuteProcess(
            cmd=[
                'ros2', 'run', 'lbr_demos_advanced_py', 'admittance_control',
                '--ros-args', '-r', '__ns:=/lbr',
                '--params-file', admittance_param
            ],
            output='screen'
        ),

        # Tac3D A1-0041L
        ExecuteProcess(
            cmd=[
                './Tac3D', '-c', 'config/A1-0041L', '-d', '2',
                '-i', '127.0.0.1', '-p', '9988'
            ],
            cwd='/home/lyj/robot_space_2/ros2_driver_layer/src/Tac3D-v3.1.3-linux',
            output='screen'
        ),

        # Tac3D A1-0040R
        ExecuteProcess(
            cmd=[
                './Tac3D', '-c', 'config/A1-0040R', '-d', '4',
                '-i', '127.0.0.1', '-p', '9989'
            ],
            cwd='/home/lyj/robot_space_2/ros2_driver_layer/src/Tac3D-v3.1.3-linux',
            output='screen'
        ),

        # 右传感器
        Node(
            package='haptic',
            executable='tac3d_r',
            name='tac3d_r_node',
            output='screen'
        ),

        # 左传感器
        Node(
            package='haptic',
            executable='tac3d_l',
            name='tac3d_l_node',
            output='screen'
        ),

        # 条件执行 reset
        ExecuteProcess(
            cmd=[
                'ros2', 'run', 'haptic', 'reset'
            ],
            condition=IfCondition(do_reset),
            output='screen'
        ),
    ])
