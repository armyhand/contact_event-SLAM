# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入必要的模块
import time
import rclpy
from rclpy.node import Node
from tutorial_interfaces.msg import Cloud
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from .PyTac3D import Sensor
from cv_bridge import CvBridge
import numpy as np


class Tac3DPublisher(Node):

    def __init__(self):
        super().__init__('tac3d_publisher_l')
        # 发布器定义
        self.publisher_positions = self.create_publisher(Cloud, 'positions_l', 10)  # P数据 (Image)
        self.publisher_displacements = self.create_publisher(Cloud, 'displacements_l', 10)  # D数据 (Image)
        self.publisher_forces = self.create_publisher(Cloud, 'forces_l', 10)  # F数据 (Image)
        self.publisher_resultant_force = self.create_publisher(Vector3, 'resultant_force_l', 10)  # Fr数据 (Vector3)
        self.publisher_resultant_moment = self.create_publisher(Vector3, 'resultant_moment_l', 10)  # Mr数据 (Vector3)
        self.publisher_index = self.create_publisher(Float32, 'index_l', 10)  # 索引数据
        
        # CV Bridge用于图像转换
        self.bridge = CvBridge()
        
        # 传感器初始化
        self.sensor = Sensor(recvCallback=self.Tac3DRecvCallback, port=9988)


    def Tac3DRecvCallback(self, frame, param):
        """处理传感器帧数据并发布相应的ROS2消息"""
        if frame is None:
            self.get_logger().warn('接收到空帧数据')
            return
            
        try:
            # 获取各种数据
            idx = frame.get('index')
            P = frame.get('3D_Positions')
            D = frame.get('3D_Displacements')
            F = frame.get('3D_Forces')
            Fr = frame.get('3D_ResultantForce')
            Mr = frame.get('3D_ResultantMoment')

            # 发布索引数据
            if idx is not None:
                self.publish_float_data(idx, 'index')
            
            # 发布位置数据 (Image)
            if P is not None:
                self.publish_cloud_data(P, 'positions', self.publisher_positions)

            # 发布位移数据 (Image)
            if D is not None:
                self.publish_cloud_data(D, 'displacements', self.publisher_displacements)

            # 发布力数据 (Image) - 发布所有400个点的力数据
            if F is not None:
                self.publish_cloud_data(F, 'forces', self.publisher_forces)

            # 发布合力数据 (Vector3)
            if Fr is not None:
                self.publish_vector3_data(Fr, 'resultant_force', self.publisher_resultant_force)
            
            # 发布合力矩数据 (Vector3)
            if Mr is not None:
                self.publish_vector3_data(Mr, 'resultant_moment', self.publisher_resultant_moment)
                
        except Exception as e:
            self.get_logger().error(f'处理帧数据时发生错误: {str(e)}')

    def publish_float_data(self, data, data_name):
        """发布Float32类型数据"""
        try:
            if data is not None:
                msg = Float32()
                msg.data = float(data)
                self.publisher_index.publish(msg)
                # self.get_logger().info(f'发布 {data_name} 数据: {msg.data}')
            else:
                self.get_logger().warn(f'未能获取到有效的 {data_name} 数据')
        except (ValueError, TypeError) as e:
            self.get_logger().error(f'发布 {data_name} 数据时类型转换错误: {str(e)}')

    def publish_vector3_data(self, data, data_name, publisher):
        """发布Vector3类型数据"""
        try:
            if data is not None:
                msg = Vector3()
                # 检查数据是否为1×3矩阵或3元素数组
                if hasattr(data, 'shape') and len(data.shape) == 2:
                    # 如果是numpy数组且为1×3矩阵
                    if data.shape == (1, 3):
                        msg.x = float(data[0, 0])
                        msg.y = float(data[0, 1])
                        msg.z = float(data[0, 2])
                    else:
                        self.get_logger().warn(f'{data_name} 数据形状不正确: {data.shape}，期望 (1, 3)')
                        return
                elif hasattr(data, '__len__') and len(data) >= 3:
                    # 如果是列表或一维数组
                    msg.x = float(data[0])
                    msg.y = float(data[1])
                    msg.z = float(data[2])
                else:
                    self.get_logger().warn(f'未能获取到有效的 {data_name} 数据或数据不完整')
                    return
                
                publisher.publish(msg)
                # self.get_logger().info(f'发布 {data_name} 数据: {msg.x}, {msg.y}, {msg.z}')
            else:
                self.get_logger().warn(f'未能获取到有效的 {data_name} 数据')
        except (ValueError, TypeError, IndexError) as e:
            self.get_logger().error(f'发布 {data_name} 数据时发生错误: {str(e)}')

    def publish_cloud_data(self, data, data_name, publisher):
        """发布Cloud类型数据"""
        try:
            if data is not None and len(data) > 0:
                msg = Cloud()
                
                # 检查数据维度
                if len(data) != 400:
                    self.get_logger().warn(f'{data_name} 数据长度不是400: {len(data)}')
                
                # 初始化固定长度的数组（400个元素）
                row1 = [0.0] * 400
                row2 = [0.0] * 400
                row3 = [0.0] * 400
                
                # 填充有效数据
                valid_count = 0
                for i in range(min(len(data), 400)):
                    if data[i] is not None and len(data[i]) >= 3:
                        row1[i] = float(data[i][0])
                        row2[i] = float(data[i][1])
                        row3[i] = float(data[i][2])
                        valid_count += 1
                
                if valid_count == 0:
                    self.get_logger().warn(f'{data_name} 数据中没有有效的3D点')
                    return
                
                # 设置Cloud消息
                msg.row1 = row1
                msg.row2 = row2
                msg.row3 = row3
                
                publisher.publish(msg)
                # self.get_logger().info(f'发布 {data_name} 数据: {valid_count} 个有效点')
            else:
                self.get_logger().warn(f'未能获取到有效的 {data_name} 数据或数据为空')
        except (ValueError, TypeError, IndexError) as e:
            self.get_logger().error(f'发布 {data_name} 数据时发生错误: {str(e)}')
    
    def publish_image_data(self, data, data_name, publisher):
        """发布Image类型数据 (20x20x3 tactile force data as 32FC3)"""
        try:
            if data is not None and len(data) > 0:
                # 将一维数据转换为20x20x3的数组
                image_array = np.zeros((20, 20, 3), dtype=np.float32)
                
                # 填充数据
                for i in range(min(len(data), 400)):  # 400 = 20*20
                    if data[i] is not None and len(data[i]) >= 3:
                        row = i // 20
                        col = i % 20
                        image_array[row, col, 0] = float(data[i][0])
                        image_array[row, col, 1] = float(data[i][1])
                        image_array[row, col, 2] = float(data[i][2])
                
                # 创建ROS Image消息
                msg = self.bridge.cv2_to_imgmsg(image_array, encoding="32FC3")
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = f"tac3d_{data_name}"
                
                publisher.publish(msg)
                # self.get_logger().info(f'发布 {data_name} Image数据: shape {image_array.shape}')
            else:
                self.get_logger().warn(f'未能获取到有效的 {data_name} Image数据或数据为空')
        except Exception as e:
            self.get_logger().error(f'发布 {data_name} Image数据时发生错误: {str(e)}')

    def publish(self):
        # 等待 Tac3D-Desktop 启动传感器并建立连接
        self.sensor.waitForFrame()

        time.sleep(5)  # 等待 5 秒钟

        # 发送一次校准信号（确保传感器未与任何物体接触！）
        self.sensor.calibrate('DL1-GWM0072')

        time.sleep(5)  # 等待 5 秒钟

        # 创建一个定时器，定期检查并发布数据
        timer_period = 0.01  # 100Hz，每10ms检查一次
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        """定时器回调函数，定期检查并发布传感器数据"""
        frame = self.sensor.getFrame()
        if frame is not None:
            self.Tac3DRecvCallback(frame, None)


def main(args=None):
    rclpy.init(args=args)
    tac3d_publisher = Tac3DPublisher()
    tac3d_publisher.publish()
    rclpy.spin(tac3d_publisher)
    tac3d_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

