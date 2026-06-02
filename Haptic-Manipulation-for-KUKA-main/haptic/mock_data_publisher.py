#!/usr/bin/env python3
"""
模拟数据发布器 - 用于测试ACT Bridge
发布模拟的触觉传感器数据和位姿数据
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
import numpy as np
from cv_bridge import CvBridge
import time


class MockDataPublisher(Node):
    """模拟数据发布器"""
    
    def __init__(self):
        super().__init__('mock_data_publisher')
        
        # 发布器
        self.forces_l_pub = self.create_publisher(Image, '/forces_l', 10)
        self.forces_r_pub = self.create_publisher(Image, '/forces_r', 10)
        self.pose_pub = self.create_publisher(Pose, '/lbr/state/pose', 10)
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # 模拟位姿
        self.pose_x = 0.5
        self.pose_y = 0.0
        self.pose_z = 0.3
        
        # 定时器 - 模拟不同频率
        self.tactile_timer = self.create_timer(1.0/30.0, self.publish_tactile)  # 30Hz
        self.pose_timer = self.create_timer(1.0/100.0, self.publish_pose)      # 100Hz
        
        self.count = 0
        
        self.get_logger().info("模拟数据发布器启动")
        self.get_logger().info("  /forces_l: 30Hz")
        self.get_logger().info("  /forces_r: 30Hz")
        self.get_logger().info("  /lbr/state/pose: 100Hz")
    
    def publish_tactile(self):
        """发布触觉数据（32FC3格式）"""
        # 生成随机力数据 (20, 20, 3)
        forces = np.random.randn(20, 20, 3).astype(np.float32) * 0.1
        
        # 添加一些模式（模拟真实传感器）
        x, y = np.meshgrid(np.arange(20), np.arange(20))
        pattern = np.sin(x/5.0 + self.count*0.1) * np.cos(y/5.0 + self.count*0.1)
        forces[:, :, 0] += pattern * 0.5
        
        # 转换为Image消息
        img_msg_l = self.bridge.cv2_to_imgmsg(forces, encoding="32FC3")
        img_msg_l.header.stamp = self.get_clock().now().to_msg()
        img_msg_l.header.frame_id = "tactile_left"
        
        # 右传感器稍微不同
        forces_r = forces + np.random.randn(20, 20, 3).astype(np.float32) * 0.05
        img_msg_r = self.bridge.cv2_to_imgmsg(forces_r, encoding="32FC3")
        img_msg_r.header.stamp = self.get_clock().now().to_msg()
        img_msg_r.header.frame_id = "tactile_right"
        
        self.forces_l_pub.publish(img_msg_l)
        self.forces_r_pub.publish(img_msg_r)
        
        self.count += 1
        
        if self.count % 30 == 0:
            self.get_logger().info(f"已发布 {self.count} 帧触觉数据")
    
    def publish_pose(self):
        """发布位姿数据"""
        # 模拟轻微运动
        t = time.time()
        self.pose_x = 0.5 + 0.01 * np.sin(t * 0.5)
        self.pose_y = 0.0 + 0.01 * np.cos(t * 0.3)
        self.pose_z = 0.3 + 0.005 * np.sin(t * 0.7)
        
        pose_msg = Pose()
        pose_msg.position.x = self.pose_x
        pose_msg.position.y = self.pose_y
        pose_msg.position.z = self.pose_z
        
        # 固定姿态
        pose_msg.orientation.x = 0.0
        pose_msg.orientation.y = 1.0
        pose_msg.orientation.z = 0.0
        pose_msg.orientation.w = 0.0
        
        self.pose_pub.publish(pose_msg)


def main():
    rclpy.init()
    
    print("\n" + "="*60)
    print("模拟数据发布器")
    print("="*60)
    print("用于测试ACT Bridge节点")
    print("发布模拟的触觉和位姿数据")
    print("="*60 + "\n")
    
    node = None
    try:
        node = MockDataPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n停止发布...")
    finally:
        if node:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
