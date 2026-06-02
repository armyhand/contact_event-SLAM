# KUKA机械臂contact-event SLAM规划和真机代码

1. 安装lbr_stack(https://github.com/lbr-stack/lbr_fri_ros2_stack)

2. 复制Haptic-Manipulation-for-KUKA 于lbr-stack/src中。

3. colcon build --symlink-install

4. 其余的python文件和corresponding files复制到lbr_fri_ros2_stack/lbr_demos/lbr_demos_advanced_py/lbr_demos_advanced_py中，并修改setup.py

5. colcon build --symlink-install
