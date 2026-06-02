"""
ACT桥接节点 - Temporal Ensemble版本
ROS2 (Python 3.12) ← ZMQ → ACT Server (Python 3.8)

功能:
- 订阅触觉传感器和机器人位姿 (多频率)
- 数据同步和降采样
- 通过ZMQ与ACT服务器通信
- Temporal Ensemble加权输出
- 高频发布动作命令

使用方法:
    # Terminal 1: 启动ACT推理服务器
    conda activate act_py38
    cd /home/lyj/robot_space_2/ros2_driver_layer/TAC_ACT
    python act_inference_server.py

    # Terminal 2: 启动ROS2桥接节点
    source install/setup.zsh
    ros2 run haptic act_bridge

    # 可选参数
    ros2 run haptic act_bridge
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image
import zmq
import numpy as np
import pickle
import cv2
from cv_bridge import CvBridge
from collections import deque
import time
import threading


class ACTBridgeNode(Node):
    """ACT桥接节点 - Temporal Ensemble版本"""
    
    def __init__(self):
        super().__init__('act_bridge_node')
        
        # 配置参数
        self.server_address = 'tcp://localhost:5555'
        self.control_rate = 10.0
        self.chunk_size = 30
        self.temporal_ensemble = False
        self.k_ensemble = 10
        self.buffer_size = 100
        
        
        # ZMQ客户端
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(self.server_address)
        self.socket_lock = threading.Lock()
        
        # 数据锁（保护传感器缓冲区）
        self.data_lock = threading.Lock()
        
        # 传感器缓冲区
        self.tactile_left_buffer = deque(maxlen=self.buffer_size)
        self.tactile_right_buffer = deque(maxlen=self.buffer_size)
        self.pose_buffer = deque(maxlen=self.buffer_size)
        
        # 当前位姿（用于增量计算）
        self.current_pose_xyz = np.zeros(3, dtype=np.float32)  # [x, y, z]
        
        # Chunk缓冲区（用于temporal ensemble）
        self.chunk_buffer = deque(maxlen=self.k_ensemble)
        self.chunk_buffer_lock = threading.Lock()
        
        # 状态
        self.is_init = False
        self.is_inferring = False
        self.query_idx = 0
        self.last_inference_time = 0
        
        # 统计
        self.stats = {
            'inference_count': 0,
            'inference_times': deque(maxlen=100),
            'publish_count': 0
        }
        
        # ROS订阅器
        self.tactile_left_sub = self.create_subscription(
            Image, '/forces_l', self.tactile_left_callback, 10)
        self.tactile_right_sub = self.create_subscription(
            Image, '/forces_r', self.tactile_right_callback, 10)
        self.pose_sub = self.create_subscription(
            Pose, '/lbr/state/pose', self.pose_callback, 10)
        
        # ROS发布器
        self.action_pub = self.create_publisher(Pose, '/ab_action', 10)
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # 主控制定时器
        self.control_timer = self.create_timer(
            1.0 / self.control_rate, self.control_callback)
        
        # 统计定时器
        self.stats_timer = self.create_timer(5.0, self.print_stats)
        
        self.get_logger().info(
            f"ACT桥接节点启动:\n"
            f"  服务器: {self.server_address}\n"
            f"  控制频率: {self.control_rate}Hz\n"
            f"  Chunk大小: {self.chunk_size}\n"
            f"  Ensemble: {'启用' if self.temporal_ensemble else '禁用'} (窗口={self.k_ensemble})")
    
    def tactile_left_callback(self, msg):
        """左传感器力数据回调"""
        with self.data_lock:
            force_data = self._extract_image_data(msg)
            if force_data is not None:
                self.tactile_left_buffer.append((time.time(), force_data))
    
    def tactile_right_callback(self, msg):
        """右传感器力数据回调"""
        with self.data_lock:
            force_data = self._extract_image_data(msg)
            if force_data is not None:
                self.tactile_right_buffer.append((time.time(), force_data))
    
    
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
    
    def pose_callback(self, msg):
        """位姿回调"""
        with self.data_lock:
            pose_data = np.array([
                msg.position.x,
                msg.position.y,
                msg.position.z,
            ], dtype=np.float32)
            self.pose_buffer.append((time.time(), pose_data))
            
            # 更新当前位姿XYZ（用于增量计算）
            self.current_pose_xyz = pose_data.copy()
            
            if not self.is_init and len(self.pose_buffer) >= 10:
                self.is_init = True
                self.get_logger().info("✅ 传感器数据已初始化")
    
    def get_latest_data(self):
        """获取最新同步数据"""
        if (len(self.tactile_left_buffer) == 0 or 
            len(self.tactile_right_buffer) == 0 or 
            len(self.pose_buffer) == 0):
            return None
        
        return {
            'tactile_left': self.tactile_left_buffer[-1][1],
            'tactile_right': self.tactile_right_buffer[-1][1],
            'qpos': self.pose_buffer[-1][1],
            'timestamp': time.time()
        }
    
    def control_callback(self):
        """主控制循环"""
        if not self.is_init:
            return
        
        current_time = time.time()
        
        # 检查是否需要发起新的推理请求
        time_since_last = current_time - self.last_inference_time
        expected_interval = 1.0 / self.control_rate
        
        if time_since_last >= expected_interval and not self.is_inferring:
            data = self.get_latest_data()
            if data is not None:
                self.last_inference_time = current_time
                threading.Thread(
                    target=self._inference_thread,
                    args=(data, self.query_idx),
                    daemon=True
                ).start()
                self.query_idx += 1
        
        # 计算并发布动作
        action = self._compute_action_with_ensemble()
        if action is not None:
            self._publish_action(action)
            self.stats['publish_count'] += 1
    
    def _inference_thread(self, data, query_idx):
        """异步推理线程"""
        self.is_inferring = True
        inference_start = time.time()
        
        try:
            # tactile_left/right shape: (3, 20, 20)
            # 需要转换为(20, 20, 3)格式用于ACT server
            tactile_left = np.transpose(data['tactile_left'], (1, 2, 0)).astype(np.float32)
            tactile_right = np.transpose(data['tactile_right'], (1, 2, 0)).astype(np.float32)
            
            inference_data = {
                'tactile_left': tactile_left,  # (20, 20, 3)
                'tactile_right': tactile_right,  # (20, 20, 3)
                'qpos': data['qpos'],
                'timestamp': data['timestamp'],
                'query_idx': query_idx
            }
            
            # ZMQ推理请求
            with self.socket_lock:
                self.socket.send(pickle.dumps(inference_data))
                response = pickle.loads(self.socket.recv())
            
            if 'error' in response:
                self.get_logger().error(f"推理错误: {response['error']}")
                return
            
            # 获取actions chunk
            actions = response['actions']  # (chunk_size, action_dim)
            
            # 存入chunk缓冲区
            with self.chunk_buffer_lock:
                self.chunk_buffer.append({
                    'timestamp': time.time(),
                    'actions': actions,
                    'query_idx': query_idx
                })
            
            # 记录推理时间
            inference_time = (time.time() - inference_start) * 1000
            self.stats['inference_times'].append(inference_time)
            self.stats['inference_count'] += 1
            
        except zmq.ZMQError as e:
            self.get_logger().error(f"ZMQ通信错误: {e}")
        except Exception as e:
            self.get_logger().error(f"推理失败: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_inferring = False
    
    def _compute_action_with_ensemble(self):
        """使用Temporal Ensemble计算当前动作"""
        with self.chunk_buffer_lock:
            if len(self.chunk_buffer) == 0:
                return None
            
            if not self.temporal_ensemble:
                # 不使用ensemble，直接取最新chunk的第一个动作
                return self.chunk_buffer[-1]['actions'][0]
            
            # Temporal Ensemble加权
            predictions = []
            weights = []
            
            current_query = self.query_idx
            
            for i, chunk in enumerate(self.chunk_buffer):
                # 计算时间偏移
                time_offset = current_query - chunk['query_idx']
                
                if 0 <= time_offset < self.chunk_size:
                    action = chunk['actions'][time_offset]
                    predictions.append(action)
                    
                    # 指数衰减权重
                    age = len(self.chunk_buffer) - 1 - i
                    weight = np.exp(-age * 0.1)
                    weights.append(weight)
            
            if len(predictions) == 0:
                return self.chunk_buffer[-1]['actions'][0]
            
            # 归一化权重并加权平均
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            weighted_action = np.zeros(3)  # action_dim=3
            for pred, w in zip(predictions, weights):
                weighted_action += pred * w
            
            return weighted_action
    
    def _publish_action(self, action_pose: np.ndarray):
        """
        发布动作到/ab_action话题
        
        Args:
            action_pose: (3,) [x, y, z]
        """
        # action是delta增量，加到当前位姿上得到目标位姿
        final_action = self.current_pose_xyz + action_pose
        
        # action 是 absolute
        final_action = action_pose
        
        action_pose = Pose()
        action_pose.position.x = float(final_action[0])
        action_pose.position.y = float(final_action[1])
        action_pose.position.z = float(final_action[2])

        # 锁定姿态为 (x=0, y=1, z=0, w=0)
        action_pose.orientation.x = 0.0
        action_pose.orientation.y = 1.0
        action_pose.orientation.z = 0.0
        action_pose.orientation.w = 0.0
        
        self.action_pub.publish(action_pose)
    
    def print_stats(self):
        """打印统计信息"""
        if self.stats['inference_count'] == 0:
            return
        
        avg_inference = np.mean(self.stats['inference_times']) if len(self.stats['inference_times']) > 0 else 0
        
        self.get_logger().info(
            f"统计 [过去5秒]:\n"
            f"  推理: {self.stats['inference_count']}次, "
            f"平均: {avg_inference:.1f}ms\n"
            f"  发布: {self.stats['publish_count']}次\n"
            f"  Chunk缓冲: {len(self.chunk_buffer)}/{self.k_ensemble}",
            throttle_duration_sec=5.0)
        
        # 重置计数器
        self.stats['inference_count'] = 0
        self.stats['publish_count'] = 0


def main():
    rclpy.init()
    
    print("\n" + "="*70)
    print("ACT桥接节点 - Temporal Ensemble模式")
    print("="*70)
    print("架构:")
    print("  传感器 → 缓冲区 → ZMQ → ACT服务器 → Chunk → Ensemble → /ab_action")
    print("="*70 + "\n")
    
    node = None
    try:
        node = ACTBridgeNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n停止节点...")
    finally:
        if node:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
