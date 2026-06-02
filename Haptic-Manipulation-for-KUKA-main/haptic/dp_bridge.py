#!/usr/bin/env python3
"""
DP桥接节点 - ROS2到DP推理服务器的桥接
订阅触觉传感器和位姿数据，通过ZMQ与DP服务器通信，发布动作

数据流:
    /forces_l, /forces_r (30Hz) → 缓冲区 (10Hz采样) → ZMQ → DP服务器
    DP服务器 → 动作序列 → 动作发布 (10Hz)

节点特点:
    - 异步推理: 推理线程独立，不阻塞ROS2回调
    - 观测队列管理: 需要3帧历史观测才能推理
    - 动作序列缓冲: 一次推理产生8步动作，逐步发布
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge
import numpy as np
import zmq
import pickle
import threading
import time
from collections import deque


class DPBridgeNode(Node):
    """DP桥接节点"""
    
    def __init__(self):
        super().__init__('dp_bridge_node')
        
        # ====================================================================
        # 配置参数
        # ====================================================================
        self.zmq_server = "tcp://localhost:5556"  # DP服务器地址
        self.control_rate = 10.0  # 控制频率 (Hz)
        self.n_obs_steps = 3  # 需要的历史观测帧数
        self.n_action_steps = 8  # 每次推理产生的动作数
        
        print("\n" + "="*70)
        print("DP桥接节点")
        print("="*70)
        print("架构:")
        print("  传感器 → 缓冲区 (10Hz采样) → ZMQ → DP服务器 → 动作队列 → /ab_action")
        print("="*70)
        print()
        
        # ====================================================================
        # ROS2接口
        # ====================================================================
        self.bridge = CvBridge()
        
        # 订阅触觉传感器 (32FC3编码)
        self.sub_left = self.create_subscription(
            Image, '/forces_l', self.left_callback, 10)
        self.sub_right = self.create_subscription(
            Image, '/forces_r', self.right_callback, 10)
        
        # 订阅机器人位姿
        self.sub_pose = self.create_subscription(
            Pose, '/lbr/state/pose', self.pose_callback, 10)
        
        # 发布动作
        self.pub_action = self.create_publisher(Pose, '/ab_action', 10)
        
        # ====================================================================
        # 数据缓冲
        # ====================================================================
        # 观测队列：维护最近3帧的观测数据
        self.obs_queue = deque(maxlen=self.n_obs_steps)  # [(left, right), ...]
        self.current_pose_xyz = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.data_lock = threading.Lock()
        
        # 当前帧的临时数据
        self.latest_left = None
        self.latest_right = None
        self.last_sample_time = 0
        self.sample_interval = 1.0 / self.control_rate  # 10Hz采样
        
        # ====================================================================
        # ZMQ客户端
        # ====================================================================
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(self.zmq_server)
        self.socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5秒超时
        
        # ====================================================================
        # 状态管理
        # ====================================================================
        self.initialized = False  # 传感器数据是否初始化
        self.obs_queue_ready = False  # 观测队列是否准备好（收集了n_obs_steps帧）
        self.obs_count = 0  # 已发送的观测帧数
        
        # 动作序列管理
        self.action_queue = deque(maxlen=self.n_action_steps)  # 当前动作序列
        
        # ====================================================================
        # 统一的10Hz控制定时器
        # ====================================================================
        self.timer = self.create_timer(1.0 / self.control_rate, self._control_callback)
        
        # ====================================================================
        # 统计信息
        # ====================================================================
        self.stats_timer = self.create_timer(5.0, self._print_stats)
        self.inference_count = 0
        self.publish_count = 0
        self.inference_times = []
        
        self.get_logger().info(
            f"DP桥接节点启动:\n"
            f"  服务器: {self.zmq_server}\n"
            f"  控制频率: {self.control_rate}Hz\n"
            f"  n_obs_steps: {self.n_obs_steps}\n"
            f"  n_action_steps: {self.n_action_steps}"
        )
    
    # ========================================================================
    # ROS2回调 (高频：30Hz+，尽可能快地更新缓冲区)
    # ========================================================================
    
    def left_callback(self, msg: Image):
        """左触觉传感器回调 - 只更新最新数据"""
        with self.data_lock:
            self.latest_left = self._extract_image_data(msg)
    
    def right_callback(self, msg: Image):
        """右触觉传感器回调 - 只更新最新数据"""
        with self.data_lock:
            self.latest_right = self._extract_image_data(msg)
    
    def pose_callback(self, msg: Pose):
        """位姿回调"""
        with self.data_lock:
            self.current_pose_xyz = np.array([
                msg.position.x,
                msg.position.y,
                msg.position.z
            ], dtype=np.float32)
    
    def _extract_image_data(self, msg: Image) -> np.ndarray:
        """
        从ROS Image消息提取数据
        
        Args:
            msg: sensor_msgs/Image (encoding: 32FC3)
        
        Returns:
            np.ndarray: (20, 20, 3) float32
        """
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        return image


    # ========================================================================
    # 统一的10Hz控制循环
    # ========================================================================
    
    def _control_callback(self):
        """
        统一的10Hz控制循环，按序执行：
        1. 采样最新传感器数据到obs_queue
        2. 如果obs_queue满3帧且action_queue为空，触发推理
        3. 如果action_queue有数据，发布一个动作
        """
        # 步骤1: 采样观测数据
        with self.data_lock:
            left = self.latest_left
            right = self.latest_right
        
        # 检查数据是否就绪
        if left is not None and right is not None:
            # 采样到观测队列（长度为3，自动维护最近3帧）
            self.obs_queue.append((left.copy(), right.copy()))
            
            # 首次初始化标记
            if not self.initialized and len(self.obs_queue) >= self.n_obs_steps:
                self.initialized = True
                self.obs_queue_ready = True
                self.get_logger().info(f"✅ 观测队列已初始化 ({len(self.obs_queue)}帧)")
        
        # 步骤2: 触发推理（如果需要）
        if self.obs_queue_ready and len(self.action_queue) == 0:
            self._request_new_actions()
        
        # 步骤3: 发布动作
        if len(self.action_queue) > 0:
            action_delta = self.action_queue.popleft()
            action_pose = action_delta  # 直接使用（DP输出是绝对位姿）
            self._publish_action(action_pose)
    
    def _request_new_actions(self):
        """请求新的动作序列（发送最近3帧观测数据）"""
        try:
            # 检查obs_queue是否准备好
            if len(self.obs_queue) < self.n_obs_steps:
                return
            
            # 提取3帧数据 - 已经是 (20, 20, 3) 格式
            obs_list = list(self.obs_queue)
            tactile_left_seq = [left for left, right in obs_list]
            tactile_right_seq = [right for left, right in obs_list]
            
            # 构建请求
            request = {
                'tactile_left_seq': np.array(tactile_left_seq),  # (3, 20, 20, 3)
                'tactile_right_seq': np.array(tactile_right_seq),  # (3, 20, 20, 3)
            }
            
            start_time = time.time()
            
            self.socket.send(pickle.dumps(request))
            response = pickle.loads(self.socket.recv())
            
            inference_time = time.time() - start_time
            
            if 'error' in response:
                self.get_logger().error(f"动作推理错误: {response['error']}")
                return
            
            # 将动作序列加入队列
            actions = response['actions']  # (n_action_steps, action_dim)
            self.action_queue.clear()
            for action in actions:
                self.action_queue.append(action)
            
            self.inference_count += 1
            self.inference_times.append(inference_time * 1000)
            
        except zmq.error.Again:
            self.get_logger().warn("动作推理超时")
        except Exception as e:
            self.get_logger().error(f"动作推理错误: {e}")
    
    def _publish_action(self, action_pose: np.ndarray):
        """
        发布动作到/ab_action话题
        
        Args:
            action_pose: (3,) [x, y, z]
        """
        # action是delta增量，加到当前位姿上得到目标位姿
        final_action = self.current_pose_xyz + action_pose
        # 缩放区
        final_action_x = self.current_pose_xyz[0] + action_pose[0] 
        final_action_y = self.current_pose_xyz[1] + action_pose[1] 
        final_action_z = self.current_pose_xyz[2] + action_pose[2] 

        final_action = np.array([final_action_x, final_action_y, final_action_z], dtype=np.float32)
        
        # action 是 absolute
        # final_action = action_pose
        
        action_pose = Pose()
        action_pose.position.x = float(final_action[0])
        action_pose.position.y = float(final_action[1])
        action_pose.position.z = float(final_action[2])

        # 锁定姿态为 (x=0, y=1, z=0, w=0)
        action_pose.orientation.x = 0.0
        action_pose.orientation.y = 1.0
        action_pose.orientation.z = 0.0
        action_pose.orientation.w = 0.0
        
        self.pub_action.publish(action_pose)
        self.publish_count += 1
    
    # ========================================================================
    # 统计信息
    # ========================================================================
    
    def _print_stats(self):
        """打印统计信息"""
        if self.inference_count == 0:
            return
        
        avg_inference = np.mean(self.inference_times[-20:]) if len(self.inference_times) > 0 else 0
        queue_len = len(self.action_queue)
        
        self.get_logger().info(
            f"统计 [过去5秒]:\n"
            f"  推理: {self.inference_count}次, 平均: {avg_inference:.1f}ms\n"
            f"  发布: {self.publish_count}次\n"
            f"  观测队列: {len(self.obs_queue)}/{self.n_obs_steps}\n"
            f"  动作队列: {queue_len}/{self.n_action_steps}"
        )
        
        # 重置计数
        self.inference_count = 0
        self.publish_count = 0
    
    def destroy_node(self):
        """清理资源"""
        self.get_logger().info("停止节点...")
        self.socket.close()
        super().destroy_node()


def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    
    try:
        node = DPBridgeNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"节点错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
