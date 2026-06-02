from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'haptic'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    package_data={
        package_name: [
            'models/cnn_ae/*.py',
            'models/cnn_ae/*.pt',
            'models/feature_lstm/*.py',
            'models/feature_lstm/*.pt',
            'models/feature_lstm/*.yaml',
        ],
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'ruamel.yaml',
        'opencv-python',
        'h5py',
    ],
    zip_safe=True,
    maintainer='lyj',
    maintainer_email='lyj6626076@gmail.com',
    description='ROS2 package for Tac3D sensor and data recording',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tac3d_r = haptic.tac3d_r:main',
            'tac3d_l = haptic.tac3d_l:main',
            
            'admittance_with_zero_reset = haptic.admittance_with_zero_reset:main',
            
            'dataset_recorder = haptic.dataset_recorder:main',
            'dataset_recorder_h5 = haptic.dataset_recorder_h5:main',
            
            'test_force = haptic.test_force:main',

            'reset = haptic.reset:main',  # 重置机械臂到初始位置节点
            "execute = haptic.execute:main",  # 废弃执行器节点
            
            'cartesian_controller = haptic.cartesian_controller:main',  # 笛卡尔控制器节点
            'kb_c = haptic.keyboard_control:main',  # 键盘控制节点

            'lstm = haptic.lstm_policy:main',  # lstm策略节点（当前环境）
            'lstm_test = haptic.lstm_test:main',

            'act_bridge = haptic.act_bridge:main',  # ACT桥接节点（ZMQ客户端-act_py38环境）
            'dp_bridge = haptic.dp_bridge:main',  # DP桥接节点（ZMQ客户端-触觉图像）
            'dp_f_bridge = haptic.dp_f_bridge:main',  # DP合力桥接节点（ZMQ客户端-合力数据）
            'mock_data = haptic.mock_data_publisher:main',  # 模拟数据发布器（测试用）
        ],
    },
)

