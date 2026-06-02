import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from tutorial_interfaces.msg import Array3, Cloud
from geometry_msgs.msg import Pose, Vector3
from datetime import datetime
import threading
import uuid
import time
import h5py
import numpy as np
from collections import deque
from cv_bridge import CvBridge


class DatasetRecorderH5(Node):
    """HDF5格式数据记录器，以固定频率从数据队列中采样记录"""
    
    # 配置参数
    USE_COMPRESSION = False  # 设置为True启用数据压缩，False禁用压缩
    
    def __init__(self):
        super().__init__('dataset_recorder_h5')
        
        # 采样频率设置
        self.sampling_freq = 50.0  # Hz
        self.sampling_interval = 1.0 / self.sampling_freq
        
        # 数据压缩设置
        self.use_compression = self.USE_COMPRESSION
        
        # 创建唯一文件路径
        self.h5_file_path = self._create_unique_filepath()
        
        # 数据队列 - 存储最新的数据
        self.data_queues = {
            'joint_names': deque(maxlen=5),
            'joint_positions': deque(maxlen=5),
            'joint_velocities': deque(maxlen=5),
            'joint_efforts': deque(maxlen=5),
            'forces_l': deque(maxlen=5),
            'forces_r': deque(maxlen=5),
            'resultant_force_l': deque(maxlen=5),
            'resultant_force_r': deque(maxlen=5),
            'resultant_moment_l': deque(maxlen=5),
            'resultant_moment_r': deque(maxlen=5),
        }
        
        # 线程锁
        self.data_lock = threading.Lock()
        
        # CV Bridge用于图像转换
        self.bridge = CvBridge()
        
        # HDF5文件和数据集
        self.h5_file = None
        self.datasets = {}
        self.episode_data = []
        
        # 采样计数器
        self.frame_count = 0
        
        # 停止标志
        self.is_recording = True
        self.shutdown_requested = False
        
        # 初始化HDF5文件
        self._init_h5_file()
        
        # 创建订阅
        self._create_subscriptions()
        
        # 创建采样定时器
        self.timer = self.create_timer(self.sampling_interval, self.sample_callback)
        
        self.get_logger().info(f"HDF5数据记录器启动，采样频率: {self.sampling_freq}Hz")
        self.get_logger().info(f"数据将保存到: {self.h5_file_path}")
    
    def _create_unique_filepath(self):
        """创建唯一的HDF5文件路径"""
        current_workspace_path = os.getcwd()
        base_dir = os.path.join(current_workspace_path, 'training_data')
        
        now = datetime.now()
        timestamp = now.strftime('%Y%m%d_%H%M%S_%f')[:-3]
        process_id = os.getpid()
        unique_id = str(uuid.uuid4())[:8]
        
        filename = f"dataset_{timestamp}_{process_id}_{unique_id}.h5"
        filepath = os.path.join(base_dir, 'h5_datasets', filename)
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        return filepath
    
    def _init_h5_file(self):
        """初始化HDF5文件结构"""
        self.h5_file = h5py.File(self.h5_file_path, 'w')
        
        # 创建根组
        obs_group = self.h5_file.create_group('observation')
        action_group = self.h5_file.create_group('action')
        
        # 数据集配置表: (key, group, shape, dtype)
        dataset_specs = [
            ('joint_names',        obs_group, (0, 7),        h5py.string_dtype(encoding='utf-8')),
            ('joint_positions',    obs_group, (0, 7),        'f8'),
            ('joint_velocities',   obs_group, (0, 7),        'f8'),
            ('joint_efforts',      obs_group, (0, 7),        'f8'),
            ('forces_l',           obs_group, (0, 3, 20, 20),'f8'),
            ('forces_r',           obs_group, (0, 3, 20, 20),'f8'),
            ('resultant_force_l',  obs_group, (0, 3),        'f8'),
            ('resultant_force_r',  obs_group, (0, 3),        'f8'),
            ('resultant_moment_l', obs_group, (0, 3),        'f8'),
            ('resultant_moment_r', obs_group, (0, 3),        'f8'),
        ]
        
        # 数据压缩设置
        compression_kwargs = {'compression': 'gzip', 'compression_opts': 9} if self.use_compression else {}
        
        # 批量创建观测数据集
        for key, group, shape, dtype in dataset_specs:
            maxshape = (None,) + shape[1:]  # 第一个维度可扩展
            self.datasets[key] = group.create_dataset(
                key, shape, maxshape=maxshape, dtype=dtype, **compression_kwargs)
        
        
        # 时间戳数据集
        self.datasets['timestamps'] = self.h5_file.create_dataset(
            'timestamps', (0,), maxshape=(None,), dtype='f8',
            **compression_kwargs)
        
        # 添加元数据
        self.h5_file.attrs['sampling_frequency'] = self.sampling_freq
        self.h5_file.attrs['created_at'] = datetime.now().isoformat()
        self.h5_file.attrs['description'] = '触觉盲操作数据集 - 固定频率采样'
        self.h5_file.attrs['compression_enabled'] = self.use_compression
        self.h5_file.attrs['nan_policy'] = 'NaN值表示异常或缺失数据'
        self.h5_file.attrs['tactile_matrix_information_image_encoding'] = '32FC3'

    
    def _create_subscriptions(self):
        """创建所有ROS订阅"""
        # 订阅配置表: (message_type, topic, callback)
        subspecs = [
            (JointState, '/lbr/joint_states', self.joint_states_callback),
            (Cloud,      '/forces_l',         self.forces_l_image_callback),
            (Cloud,      '/forces_r',         self.forces_r_image_callback),
            (Vector3,     '/resultant_force_l',self.resultant_force_l_callback),
            (Vector3,     '/resultant_force_r',self.resultant_force_r_callback),
            (Vector3,     '/resultant_moment_l',self.resultant_moment_l_callback),
            (Vector3,     '/resultant_moment_r',self.resultant_moment_r_callback),
        ]
        
        # 批量创建订阅
        self.subscriptions = []
        for msg_type, topic, callback in subspecs:
            sub = self.create_subscription(msg_type, topic, callback, 10)
            self.subscriptions.append(sub)
    
    def joint_states_callback(self, msg):
        """关节状态回调"""
        with self.data_lock:
            # 提取关节名称（通常只有第一次需要）
            if msg.name:
                joint_names = list(msg.name[:7])  # 取前7个关节名称
                # 填充到7个元素
                while len(joint_names) < 7:
                    joint_names.append(f'joint_{len(joint_names)}')
                self.data_queues['joint_names'].append(joint_names)
            
            # 提取关节位置
            if msg.position:
                positions = np.array(msg.position[:7])
                if len(positions) < 7:
                    # 填充到7个元素
                    padded_positions = np.zeros(7)
                    padded_positions[:len(positions)] = positions
                    positions = padded_positions
                self.data_queues['joint_positions'].append(positions)
            
            # 提取关节速度
            if msg.velocity:
                velocities = np.array(msg.velocity[:7])
                if len(velocities) < 7:
                    # 填充到7个元素
                    padded_velocities = np.zeros(7)
                    padded_velocities[:len(velocities)] = velocities
                    velocities = padded_velocities
                self.data_queues['joint_velocities'].append(velocities)
            else:
                self.data_queues['joint_velocities'].append(np.zeros(7))
            
            # 提取关节力矩
            if msg.effort:
                efforts = np.array(msg.effort[:7])
                if len(efforts) < 7:
                    # 填充到7个元素
                    padded_efforts = np.zeros(7)
                    padded_efforts[:len(efforts)] = efforts
                    efforts = padded_efforts
                self.data_queues['joint_efforts'].append(efforts)
            else:
                self.data_queues['joint_efforts'].append(np.zeros(7))
    
    def forces_l_image_callback(self, msg):
        """左传感器力图像回调"""
        with self.data_lock:
            force_data = self._extract_image_data(msg)
            self.data_queues['forces_l'].append(force_data)
    
    def forces_r_image_callback(self, msg):
        """右传感器力图像回调"""
        with self.data_lock:
            force_data = self._extract_image_data(msg)
            self.data_queues['forces_r'].append(force_data)
    
    def _update_xyz_queue(self, msg, queue_key):
        """通用的三维数据队列更新函数"""
        with self.data_lock:
            xyz_data = np.array([msg.x, msg.y, msg.z])
            self.data_queues[queue_key].append(xyz_data)
    
    def resultant_force_l_callback(self, msg):
        """左传感器合力回调"""
        self._update_xyz_queue(msg, 'resultant_force_l')
    
    def resultant_force_r_callback(self, msg):
        """右传感器合力回调"""
        self._update_xyz_queue(msg, 'resultant_force_r')
    
    def resultant_moment_l_callback(self, msg):
        """左传感器合力矩回调"""
        self._update_xyz_queue(msg, 'resultant_moment_l')
    
    def resultant_moment_r_callback(self, msg):
        """右传感器合力矩回调"""
        self._update_xyz_queue(msg, 'resultant_moment_r')
    
    def _extract_image_data(self, msg):
        """从Image消息提取20x20x3的力数据
        
        使用标准的sensor_msgs/Image消息类型，32FC3编码。
        这是ROS2中处理多维数据的标准做法。
        """
        try:
            if msg.encoding == "32FC3":
                # 将Image消息转换为numpy数组
                force_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC3")
                # force_image shape: (20, 20, 3)
                # 转换为 (3, 20, 20) 格式
                force_data = np.transpose(force_image, (2, 0, 1))
                return self._clean_array(force_data)
            else:
                self.get_logger().warn(f"不支持的图像编码: {msg.encoding}, 期望 32FC3")
                return np.full((3, 20, 20), np.nan)
        except Exception as e:
            self.get_logger().error(f"图像转换错误: {e}")
            return np.full((3, 20, 20), np.nan)
    
    def _clean_array(self, data):
        """清理数组中的异常值，保留NaN和无穷大值以表示异常数据"""
        data = np.array(data, dtype=np.float64)
        # 处理绝对值过大的数据，将其设为NaN
        data[np.abs(data) > 1e8] = np.nan
        # 保持原有的NaN和无穷大值不变
        return data
    
    def _get_latest_data(self, key, default_value):
        """从队列获取最新数据"""
        if self.data_queues[key]:
            return self.data_queues[key][-1]
        else:
            if isinstance(default_value, list):
                return default_value  # 返回字符串列表
            else:
                return np.zeros(default_value)  # 返回零数组
    
    def sample_callback(self):
        """采样回调 - 以固定频率采集数据"""
        # 检查是否应该停止记录
        if not self.is_recording or self.shutdown_requested:
            return
            
        try:
            current_time = time.time()
            
            with self.data_lock:
                # 获取当前时刻的所有数据
                sample_data = {
                    'joint_names': self._get_latest_data('joint_names', ['joint_0', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']),
                    'joint_positions': self._get_latest_data('joint_positions', (7,)),
                    'joint_velocities': self._get_latest_data('joint_velocities', (7,)),
                    'joint_efforts': self._get_latest_data('joint_efforts', (7,)),
                    'forces_l': self._get_latest_data('forces_l', (3, 20, 20)),
                    'forces_r': self._get_latest_data('forces_r', (3, 20, 20)),
                    'resultant_force_l': self._get_latest_data('resultant_force_l', (3,)),
                    'resultant_force_r': self._get_latest_data('resultant_force_r', (3,)),
                    'resultant_moment_l': self._get_latest_data('resultant_moment_l', (3,)),
                    'resultant_moment_r': self._get_latest_data('resultant_moment_r', (3,)),
                    'timestamp': current_time * 1000,  # 转换为毫秒
                }
                
                
                # 添加到episode数据
                self.episode_data.append(sample_data)
                self.frame_count += 1
                
                # 写入频率，每100帧写入一次
                if self.frame_count % 100 == 0:
                    self._write_to_h5()
                    self.get_logger().info(f"已采样 {self.frame_count} 帧数据")
                
        except Exception as e:
            self.get_logger().error(f"采样过程中发生错误: {e}")
            # 发生严重错误时停止记录
            self.is_recording = False
    
    def _write_to_h5(self):
        """将缓存数据写入HDF5文件"""
        if not self.episode_data or not self.h5_file:
            return
            
        try:
        
            # 准备数据数组
            n_samples = len(self.episode_data)
            
            # 扩展数据集大小
            for key, dataset in self.datasets.items():
                old_size = dataset.shape[0]
                new_size = old_size + n_samples
                dataset.resize((new_size,) + dataset.shape[1:])
            
            # 写入数据
            start_idx = self.datasets['joint_positions'].shape[0] - n_samples
            
            for i, sample in enumerate(self.episode_data):
                idx = start_idx + i
                
                self.datasets['joint_names'][idx] = sample['joint_names']
                self.datasets['joint_positions'][idx] = sample['joint_positions']
                self.datasets['joint_velocities'][idx] = sample['joint_velocities']
                self.datasets['joint_efforts'][idx] = sample['joint_efforts']
                self.datasets['forces_l'][idx] = sample['forces_l']
                self.datasets['forces_r'][idx] = sample['forces_r']
                self.datasets['resultant_force_l'][idx] = sample['resultant_force_l']
                self.datasets['resultant_force_r'][idx] = sample['resultant_force_r']
                self.datasets['resultant_moment_l'][idx] = sample['resultant_moment_l']
                self.datasets['resultant_moment_r'][idx] = sample['resultant_moment_r']
                self.datasets['timestamps'][idx] = sample['timestamp']
        
            # 刷新到磁盘
            self.h5_file.flush()
            
            # 清空缓存
            self.episode_data.clear()
            
        except Exception as e:
            self.get_logger().error(f"写入HDF5文件时发生错误: {e}")
            # 发生写入错误时，不清空缓存，等待下次重试
            raise
    
    def finalize_recording(self):
        """结束记录，写入剩余数据并关闭文件"""
        self.shutdown_requested = True
        self.is_recording = False
        
        try:
            # 停止定时器
            if hasattr(self, 'timer') and self.timer:
                self.timer.cancel()
                
            # 写入剩余数据
            if self.episode_data and self.h5_file:
                self.get_logger().info(f"正在写入剩余的 {len(self.episode_data)} 帧数据...")
                self._write_to_h5()
            
            # 更新元数据和关闭文件
            if self.h5_file:
                try:
                    total_frames = self.datasets['timestamps'].shape[0]
                    self.h5_file.attrs['total_frames'] = total_frames
                    self.h5_file.attrs['duration_seconds'] = total_frames / self.sampling_freq
                    self.h5_file.attrs['finalized_at'] = datetime.now().isoformat()
                    
                    self.get_logger().info(f"记录完成，共 {total_frames} 帧数据")
                    self.get_logger().info(f"数据已保存到: {self.h5_file_path}")
                    
                except Exception as e:
                    self.get_logger().error(f"更新元数据时发生错误: {e}")
                
                finally:
                    # 确保文件被关闭
                    try:
                        self.h5_file.close()
                    except:
                        pass
                    self.h5_file = None
                    
        except Exception as e:
            self.get_logger().error(f"结束记录时发生错误: {e}")
        
        finally:
            # 清理资源
            self.episode_data.clear()
            if hasattr(self, 'datasets'):
                self.datasets.clear()
    
    def destroy_node(self):
        """销毁节点"""
        self.get_logger().info("正在停止数据记录器...")
        
        try:
            self.finalize_recording()
        except Exception as e:
            self.get_logger().error(f"销毁节点时发生错误: {e}")
        finally:
            super().destroy_node()
            self.get_logger().info("数据记录器已停止")


def main(args=None):
    rclpy.init(args=args)
    
    print("启动HDF5数据记录器...")
    print("功能: 以固定频率采样记录各种传感器数据到HDF5格式")
    print()
    
    recorder = None
    try:
        recorder = DatasetRecorderH5()
        print(f"数据将保存到: {recorder.h5_file_path}")
        print(f"数据压缩: {'启用' if recorder.use_compression else '禁用'}")
        print("按 Ctrl+C 停止记录")
        print("提示: 修改 USE_COMPRESSION = True 启用数据压缩")
        print()
        
        rclpy.spin(recorder)
        
    except KeyboardInterrupt:
        print("\n收到停止信号，正在安全关闭...")
    except Exception as e:
        print(f"\n运行时发生错误: {e}")
    finally:
        print("正在清理资源...")
        if recorder:
            try:
                recorder.destroy_node()
            except Exception as e:
                print(f"清理节点时发生错误: {e}")
        
        try:
            rclpy.shutdown()
        except:
            pass
        
        print("程序已退出")


if __name__ == '__main__':
    main()

# dataset.h5
# ├── observation/
# │   ├── joint_names (N, 7) - 字符串，关节名称
# │   ├── joint_positions (N, 7) - 关节位置
# │   ├── joint_velocities (N, 7) - 关节速度  
# │   ├── joint_efforts (N, 7) - 关节力矩
# │   ├── forces_l (N, 3, 20, 20) - 左手触觉力阵列（来自Image 32FC3）
# │   ├── forces_r (N, 3, 20, 20) - 右手触觉力阵列（来自Image 32FC3）
# │   ├── resultant_force_l (N, 3) - 左手合力
# │   ├── resultant_force_r (N, 3) - 右手合力
# │   ├── resultant_moment_l (N, 3) - 左手合力矩
# │   └── resultant_moment_r (N, 3) - 右手合力矩
# ├── timestamps (N,) - 时间戳(ms)
# └── 元数据:
#     ├── sampling_frequency: 50.0
#     ├── compression_enabled: true/false
#     ├── nan_policy: "NaN值表示异常或缺失数据"
#     └── force_image_encoding: "32FC3" - 力数据使用Image消息类型
