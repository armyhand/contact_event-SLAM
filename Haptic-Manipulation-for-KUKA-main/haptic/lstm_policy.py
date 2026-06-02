"""
LSTM策略节点 - 基于触觉数据的神经网络策略
实时订阅触觉传感器数据，通过LSTM网络推理，发布动作增量
"""

import os
import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Pose, Point, Quaternion
from tutorial_interfaces.msg import Array3
import threading
import time
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from cv_bridge import CvBridge
import yaml
from scipy.spatial.transform import Rotation as R

try:
    from tf2_ros import TransformException
    from tf2_ros.buffer import Buffer
    from tf2_ros.transform_listener import TransformListener
    import tf2_geometry_msgs
except ImportError:
    print("Warning: tf2_ros not available, will use alternative FK method")


# 导入LSTM模型
try:
    from haptic.models.feature_lstm.feature_lstm import TactilePolicyFeatureLSTM
except ImportError:
    # 开发环境下的导入
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'feature_lstm')
    sys.path.insert(0, model_path)
    from feature_lstm import TactilePolicyFeatureLSTM


class LSTMPolicyNode(Node):
    """LSTM策略节点 - 实时推理触觉数据"""
    
    def __init__(self):
        super().__init__('lstm_policy_node')
        
        # 模型路径配置
        self.encoder_model_path = '/home/lyj/robot_space_2/ros2_driver_layer/src/haptic/haptic/models/cnn_ae/best_model.pt'
        self.model_dir = os.path.join(os.path.dirname(__file__), 'models', 'feature_lstm')
        self.model_path = os.path.join(self.model_dir, 'best_model.pt')
        self.config_path = os.path.join(self.model_dir, 'config.yaml')
        
        # 加载配置
        self._load_config()
        
        # 推理频率设置
        self.inference_freq = 10.0  # Hz
        self.inference_interval = 1.0 / self.inference_freq
        
        # 序列长度（与训练时一致）
        self.sequence_length = self.config['data']['sequence_length']
        
        # 数据队列 - 存储时序数据
        self.force_l_queue = deque(maxlen=self.sequence_length)
        self.force_r_queue = deque(maxlen=self.sequence_length)
        self.pose_xyz_queue = deque(maxlen=self.sequence_length)  # 位姿XYZ序列
        
        # 当前位姿（用于网络输入）
        self.current_pose_xyz = np.zeros(3, dtype=np.float32)  # [x, y, z]
        
        # 关节位置和笛卡尔位姿
        self.joint_positions = np.zeros(7, dtype=np.float32)  # 保留但不使用
        self.current_pose = None  # 当前笛卡尔位置（Pose消息）
        
        # 锁定的目标姿态 (x=0, y=1, z=0, w=0)
        self.target_orientation = Quaternion(x=0.0, y=1.0, z=0.0, w=0.0)
        
        # 增量放大因子 (临时调试用)
        self.delta_scale_x = 1.0  # X方向放大倍数
        self.delta_scale_y = 2.0  # Y方向放大倍数
        self.delta_scale_z = 1.0  # Z方向放大倍数
        
        # 线程锁
        self.data_lock = threading.Lock()
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"使用设备: {self.device}")
        
        # 加载模型
        self._load_model()
        
        # 创建订阅
        self._create_subscriptions()
        
        # 创建发布器
        self.pose_publisher = self.create_publisher(Pose, '/ab_action', 10)
        
        # 创建推理定时器
        self.inference_timer = self.create_timer(self.inference_interval, self.inference_callback)
        
        # 统计信息
        self.inference_count = 0
        self.last_log_time = time.time()
        
        self.get_logger().info(f"LSTM策略节点启动，推理频率: {self.inference_freq}Hz")
        self.get_logger().info(f"序列长度: {self.sequence_length}")
        self.get_logger().info(f"增量放大因子: DX={self.delta_scale_x}, DY={self.delta_scale_y}, DZ={self.delta_scale_z}")
    
    def _load_config(self):
        """加载配置文件"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # 提取关键配置
        self.config = {
            'model': config['model']['value'],
            'data': config['data']['value'],
        }
        
        # 归一化参数 - 从computed_normalization_params中读取
        computed_norm = config.get('computed_normalization_params', {}).get('value', {})
        
        # 如果computed参数不存在,尝试从data.normalization_config读取
        if not computed_norm:
            computed_norm = self.config['data'].get('normalization_config', {})
        
        # 构建归一化参数字典(兼容原有代码结构)
        self.norm_params = {
            'forces': {
                'method': computed_norm.get('forces', {}).get('method', 'zscore'),
                'params': {
                    'mean': computed_norm.get('forces', {}).get('params', {}).get('mean', 0.0),
                    'std': computed_norm.get('forces', {}).get('params', {}).get('std', 1.0),
                }
            },
            'action': {
                'method': computed_norm.get('actions', {}).get('method', 'zscore'),
                'params': {
                    'mean': computed_norm.get('actions', {}).get('params', {}).get('mean', 0.0),
                    'std': computed_norm.get('actions', {}).get('params', {}).get('std', 1.0),
                }
            }
        }
        
        self.get_logger().info("配置文件加载成功")
        self.get_logger().info(f"模型配置: {self.config['model']}")
        self.get_logger().info(f"归一化参数 - forces: mean={self.norm_params['forces']['params']['mean']:.6f}, std={self.norm_params['forces']['params']['std']:.6f}")
        self.get_logger().info(f"归一化参数 - action: mean={self.norm_params['action']['params']['mean']:.6f}, std={self.norm_params['action']['params']['std']:.6f}")
            
    
    def _load_model(self):
        """加载训练好的LSTM模型"""
        try:
            # 创建模型实例
            model_config = self.config['model']
            
            # 使用硬编码的绝对路径指向源码目录的CNN编码器
            cnn_encoder_path = self.encoder_model_path
            
            self.model = TactilePolicyFeatureLSTM(
                feature_dim=model_config['feature_dim'],
                action_dim=model_config['action_dim'],
                lstm_hidden_dim=model_config['lstm_hidden_dim'],
                lstm_num_layers=model_config['lstm_num_layers'],
                dropout_rate=model_config['dropout_rate'],
                pretrained_encoder_path=cnn_encoder_path,  # 使用绝对路径
                action_embed_dim=model_config['action_embed_dim'],
                fc_hidden_dims=model_config['fc_hidden_dims'],
            )
            
            # 加载权重 (PyTorch 2.6+ 需要设置 weights_only=False)
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.get_logger().info(f"加载模型checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            
            self.get_logger().info(f"模型加载成功: {self.model_path}")
            
        except Exception as e:
            self.get_logger().error(f"加载模型失败: {e}")
            raise
    
    def _create_subscriptions(self):
        """创建ROS2订阅"""
        # 订阅触觉传感器图像数据
        self.sub_forces_l = self.create_subscription(
            Image, '/forces_l', self.forces_l_callback, 10)
        
        self.sub_forces_r = self.create_subscription(
            Image, '/forces_r', self.forces_r_callback, 10)
        
        # 订阅关节状态
        self.sub_joint_states = self.create_subscription(
            JointState, '/lbr/joint_states', self.joint_states_callback, 10)
        
        # 订阅当前位姿
        self.sub_current_pose = self.create_subscription(
            Pose, '/lbr/state/pose', self.current_pose_callback, 10)
        
        self.get_logger().info("订阅创建完成")
    
    def forces_l_callback(self, msg):
        """左传感器力数据回调"""
        with self.data_lock:
            force_data = self._extract_image_data(msg)
            if force_data is not None:
                self.force_l_queue.append(force_data)
    
    def forces_r_callback(self, msg):
        """右传感器力数据回调"""
        with self.data_lock:
            force_data = self._extract_image_data(msg)
            if force_data is not None:
                self.force_r_queue.append(force_data)
    
    def joint_states_callback(self, msg):
        """关节状态回调（保留但不使用）"""
        with self.data_lock:
            if msg.position and len(msg.position) >= 7:
                self.joint_positions = np.array(msg.position[:7], dtype=np.float32)
    
    def current_pose_callback(self, msg):
        """当前位姿回调，更新当前XYZ位置"""
        with self.data_lock:
            self.current_pose = msg
            # 提取XYZ位置用于网络输入
            self.current_pose_xyz[0] = msg.position.x
            self.current_pose_xyz[1] = msg.position.y
            self.current_pose_xyz[2] = msg.position.z
            
            # 添加到位姿队列
            self.pose_xyz_queue.append(self.current_pose_xyz.copy())
    
    def _extract_image_data(self, msg):
        """从Image消息提取20x20x3的力数据"""
        try:
            if msg.encoding == "32FC3":
                force_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC3")
                # force_image shape: (20, 20, 3)
                # 转换为 (3, 20, 20) 格式
                force_data = np.transpose(force_image, (2, 0, 1)).astype(np.float32)
                return force_data
            else:
                self.get_logger().warn(f"不支持的图像编码: {msg.encoding}")
                return None
        except Exception as e:
            self.get_logger().error(f"图像转换错误: {e}")
            return None
    
    def _normalize_forces(self, forces):
        """归一化触觉力数据 (Z-score)"""
        mean = self.norm_params['forces']['params']['mean']
        std = self.norm_params['forces']['params']['std']
        return (forces - mean) / std
    
    def _normalize_pose(self, pose_xyz):
        """归一化位姿XYZ数据 (Z-score)"""
        mean = self.norm_params['action']['params']['mean']
        std = self.norm_params['action']['params']['std']
        return (pose_xyz - mean) / std
    
    def _denormalize_pose(self, normalized_pose):
        """反归一化位姿XYZ数据"""
        mean = self.norm_params['action']['params']['mean']
        std = self.norm_params['action']['params']['std']
        return normalized_pose * std + mean
    
    def inference_callback(self):
        """推理回调 - 以10Hz频率执行"""
        with self.data_lock:
            # 检查数据是否准备好
            if len(self.force_l_queue) < self.sequence_length or \
               len(self.force_r_queue) < self.sequence_length or \
               len(self.pose_xyz_queue) < self.sequence_length:
                return
            
            try:
                # 准备输入数据
                forces_l_seq = np.array(list(self.force_l_queue), dtype=np.float32)  # [seq_len, 3, 20, 20]
                forces_r_seq = np.array(list(self.force_r_queue), dtype=np.float32)  # [seq_len, 3, 20, 20]
                pose_xyz_seq = np.array(list(self.pose_xyz_queue), dtype=np.float32)  # [seq_len, 3]
                
                # 归一化力数据
                forces_l_seq = self._normalize_forces(forces_l_seq)
                forces_r_seq = self._normalize_forces(forces_r_seq)
                
                # 归一化位姿序列
                pose_xyz_seq_norm = self._normalize_pose(pose_xyz_seq)
                
                # 检查当前位姿是否可用
                if self.current_pose is None:
                    return
                
                # 转换为张量并添加batch维度
                forces_l_tensor = torch.from_numpy(forces_l_seq).unsqueeze(0).to(self.device)  # [1, seq, 3, 20, 20]
                forces_r_tensor = torch.from_numpy(forces_r_seq).unsqueeze(0).to(self.device)  # [1, seq, 3, 20, 20]
                pose_xyz_tensor = torch.from_numpy(pose_xyz_seq_norm).unsqueeze(0).to(self.device)  # [1, seq, 3]
                
                # 模型推理
                with torch.no_grad():
                    # 预测绝对位姿（归一化空间）
                    predicted_pose_norm = self.model(forces_l_tensor, forces_r_tensor, pose_xyz_tensor)
                    predicted_pose_norm = predicted_pose_norm.squeeze(0).cpu().numpy()  # [3]
                    
                    # 反归一化得到预测位姿
                    predicted_pose_xyz = self._denormalize_pose(predicted_pose_norm)
                    
                    # 计算增量 (预测位姿 - 当前位姿)
                    delta_xyz = predicted_pose_xyz - self.current_pose_xyz
                    
                    # 应用放大因子到增量
                    delta_xyz_scaled = np.array([
                        delta_xyz[0] * self.delta_scale_x,
                        delta_xyz[1] * self.delta_scale_y,
                        delta_xyz[2] * self.delta_scale_z
                    ], dtype=np.float32)
                    
                    # 最终位姿 = 当前位姿 + 放大后的增量
                    final_pose_xyz = self.current_pose_xyz + delta_xyz_scaled
                    
                    # 发布最终位姿命令
                    self._publish_pose_command(final_pose_xyz, delta_xyz, delta_xyz_scaled)
                
                # 统计信息
                self.inference_count += 1
                current_time = time.time()
                if current_time - self.last_log_time > 5.0:
                    actual_freq = self.inference_count / (current_time - self.last_log_time)
                    
                    self.get_logger().info(
                        f"推理频率: {actual_freq:.2f}Hz, "
                        f"当前位姿: [{self.current_pose_xyz[0]:.6f}, {self.current_pose_xyz[1]:.6f}, {self.current_pose_xyz[2]:.6f}], "
                        f"原始增量: [{delta_xyz[0]:.6f}, {delta_xyz[1]:.6f}, {delta_xyz[2]:.6f}], "
                        f"放大增量: [{delta_xyz_scaled[0]:.6f}, {delta_xyz_scaled[1]:.6f}, {delta_xyz_scaled[2]:.6f}], "
                        f"最终位姿: [{final_pose_xyz[0]:.6f}, {final_pose_xyz[1]:.6f}, {final_pose_xyz[2]:.6f}]"
                    )
                    self.inference_count = 0
                    self.last_log_time = current_time
                
            except Exception as e:
                self.get_logger().error(f"推理过程中发生错误: {e}")
    
    def _publish_pose_command(self, final_pose_xyz):
        """发布最终位姿命令到/ab_action"""
        # 创建新的Pose消息
        new_pose = Pose()
        
        # 发布处理后的最终位姿（绝对坐标XYZ）
        new_pose.position.x = float(final_pose_xyz[0])
        new_pose.position.y = float(final_pose_xyz[1])
        new_pose.position.z = float(final_pose_xyz[2])
        # 锁定姿态为 (x=0, y=1, z=0, w=0)
        new_pose.orientation = self.target_orientation
        
        # 发布到/ab_action话题（下一个节点会做预处理）
        self.pose_publisher.publish(new_pose)
            
    
    def destroy_node(self):
        """销毁节点"""
        self.get_logger().info("正在停止LSTM策略节点...")
        
        # 停止定时器
        if hasattr(self, 'inference_timer'):
            self.inference_timer.cancel()
        
        super().destroy_node()
        self.get_logger().info("LSTM策略节点已停止")


def main(args=None):
    rclpy.init(args=args)
    
    print("=== LSTM策略节点 ===")
    print("功能: 实时订阅触觉数据，通过LSTM网络推理，发布动作增量")
    print()
    
    node = None
    try:
        node = LSTMPolicyNode()
        print("策略节点已启动，开始推理...")
        print("按 Ctrl+C 停止")
        print()
        
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        print("\n收到停止信号...")
    except Exception as e:
        print(f"\n运行时发生错误: {e}")
    finally:
        if node:
            try:
                node.destroy_node()
            except Exception as e:
                print(f"清理节点时发生错误: {e}")
        
        try:
            rclpy.shutdown()
        except:
            pass
        
        print("程序已退出")


if __name__ == '__main__':
    main()
