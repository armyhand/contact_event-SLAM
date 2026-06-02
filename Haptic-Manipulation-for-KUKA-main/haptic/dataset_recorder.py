import os
import rclpy
from rclpy.node import Node
from lbr_fri_idl.msg import LBRWrenchCommand, LBRState
from sensor_msgs.msg import JointState, Image
from tutorial_interfaces.msg import Array3, Cloud
from std_msgs.msg import Float32
from geometry_msgs.msg import Pose, Vector3
from datetime import datetime
import csv
import threading
import uuid
import time
from cv_bridge import CvBridge
import numpy as np

class DatasetRecorder(Node):
    """简单的数据记录器，记录各个话题的数据和到达时间戳"""
    def __init__(self):
        super().__init__('dataset_recorder')
        
        # 创建唯一目录结构
        current_workspace_path = os.getcwd()
        base_dir = os.path.join(current_workspace_path, 'writting')
        
        now = datetime.now()
        timestamp = now.strftime('%Y%m%d_%H%M%S_%f')[:-3]  # 包含毫秒
        process_id = os.getpid()
        unique_id = str(uuid.uuid4())[:8]
        unique_dir_name = f"{timestamp}_{process_id}_{unique_id}"
        
        self.unique_dir = os.path.join(base_dir, '王立东', unique_dir_name)
        os.makedirs(self.unique_dir, exist_ok=True)
        
        # 线程锁，确保文件写入安全
        self.file_lock = threading.Lock()
    
        # CV Bridge用于图像转换
        self.bridge = CvBridge()
        
        # 话题类型定义
        self.topic_types = {
            # 机器人相关话题
            #'/lbr/command/wrench': 'lbr_fri_idl/msg/LBRWrenchCommand',
            'lbr/state/pose': 'geometry_msgs/msg/Pose',
            #'/lbr/state': 'lbr_fri_idl/msg/LBRState',
            '/lbr/joint_states': 'sensor_msgs/msg/JointState',
            
            # Tac3D传感器话题 - 使用标准Image消息类型
            '/positions_r': 'tutorial_interfaces/msg/Cloud',
            '/displacements_r': 'tutorial_interfaces/msg/Cloud',
            '/forces_r': 'tutorial_interfaces/msg/Cloud',
            '/resultant_force_r': 'geometry_msgs/msg/Vector3',
            '/resultant_moment_r': 'geometry_msgs/msg/Vector3',
            '/index_r': 'std_msgs/msg/Float32',
            
            '/positions_l': 'tutorial_interfaces/msg/Cloud',
            '/displacements_l': 'tutorial_interfaces/msg/Cloud',
            '/forces_l': 'tutorial_interfaces/msg/Cloud',
            '/resultant_force_l': 'geometry_msgs/msg/Vector3',
            '/resultant_moment_l': 'geometry_msgs/msg/Vector3',
            '/index_l': 'std_msgs/msg/Float32'
        }
        
        # 为每个话题创建数据文件 - 全部使用txt格式
        self.data_files = {}
        
        for topic_name, topic_type in self.topic_types.items():
            # 创建安全的文件名
            safe_topic_name = topic_name.replace('/', '_').replace(':', '_')
            
            # 所有文件都使用txt格式
            file_path = os.path.join(self.unique_dir, f'{safe_topic_name}.txt')
            self.data_files[topic_name] = open(file_path, 'w')
        
        # 创建订阅
        self._subscriptions = []
        
        # 为每个话题创建订阅
        for topic_name, topic_type in self.topic_types.items():
            msg_type = eval(topic_type.split('/')[-1])
            
            self._subscriptions.append(self.create_subscription(
                msg_type,
                topic_name,
                self.create_callback(topic_name),
                10
            ))
        
        self.get_logger().info(f"数据记录器已启动，数据将保存到: {self.unique_dir}")
        self.get_logger().info(f"监控 {len(self.topic_types)} 个话题")
        
    def create_callback(self, topic_name):
        """为指定话题创建回调函数"""
        def callback(msg):
            receive_time = self.get_clock().now().nanoseconds
            
            # 获取消息时间戳
            header_time = None
            if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
                header_time = msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec
            
            # 根据消息类型使用不同的写入方式
            topic_type = self.topic_types[topic_name]
            
            with self.file_lock:
                if 'Image' in topic_type:
                    # Image类型写入txt格式
                    self.write_image_data(topic_name, msg, receive_time, header_time)
                elif 'Cloud' in topic_type:
                    # Cloud类型写入txt格式
                    self.write_cloud_data(topic_name, msg, receive_time, header_time)
                elif 'Vector3' in topic_type:
                    # Array3类型写入txt格式
                    self.write_array3_data(topic_name, msg, receive_time)
                else:
                    # 其他类型写入txt格式
                    self.write_other_data(topic_name, msg, receive_time)
                
        return callback
    
    def write_array3_data(self, topic_name, msg, receive_time):
        """写入Array3类型的数据到txt文件 - 格式: 时间戳 换行 X 换行 Y 换行 Z 换行 30个* 换行"""
        try:
            file_handle = self.data_files[topic_name]
            
            # 写入时间戳
            file_handle.write(f"time: {receive_time/1e9}\n")
            
            # 提取并清理XYZ数据
            if hasattr(msg, 'x') and hasattr(msg, 'y') and hasattr(msg, 'z'):
                x_val = self.clean_single_float(msg.x)
                y_val = self.clean_single_float(msg.y)
                z_val = self.clean_single_float(msg.z)
                
                # 写入X, Y, Z，每个数据单独一行
                file_handle.write(f"{x_val:.6f}\n")
                file_handle.write(f"{y_val:.6f}\n")
                file_handle.write(f"{z_val:.6f}\n")
            else:
                # 无效数据时写入-99
                file_handle.write("-99.000000\n")
                file_handle.write("-99.000000\n")
                file_handle.write("-99.000000\n")
            
            # 写入30个*号作为分隔符
            file_handle.write("*" * 30 + "\n")
            file_handle.flush()
            
        except Exception as e:
            # 出错时写入错误信息
            file_handle = self.data_files[topic_name]
            file_handle.write(f"time: {receive_time}\n")
            file_handle.write("-99.000000\n")
            file_handle.write("-99.000000\n") 
            file_handle.write("-99.000000\n")
            file_handle.write("*" * 30 + "\n")
            file_handle.flush()

    def write_other_data(self, topic_name, msg, receive_time):
        """写入其他类型数据到txt文件"""
        try:
            file_handle = self.data_files[topic_name]
            
            # 写入时间戳
            file_handle.write(f"time: {receive_time/1e9}\n")
            
            # 根据消息类型提取数据
            # if hasattr(msg, 'joint_position') and hasattr(msg, 'wrench'):
            #     # LBRWrenchCommand类型 - 7个关节位置 + 6个力/力矩值
            #     # 先写入7个关节位置
            #     for pos in msg.joint_position:
            #         cleaned_pos = self.clean_single_float(pos)
            #         file_handle.write(f"{cleaned_pos:.6f}\n")
                
            #     # 写入分隔换行
            #     file_handle.write("\n")
                
            #     # 再写入6个扭矩值
            #     for wrench_val in msg.wrench:
            #         cleaned_wrench = self.clean_single_float(wrench_val)
            #         file_handle.write(f"{cleaned_wrench:.6f}\n")
            if hasattr(msg, 'position') and hasattr(msg, 'orientation'):
                # Pose类型
                pos = msg.position
                ori = msg.orientation
                file_handle.write("position: \n")
                file_handle.write(f"{self.clean_single_float(pos.x)}\n")
                file_handle.write(f"{self.clean_single_float(pos.y)}\n")
                file_handle.write(f"{self.clean_single_float(pos.z)}\n")
                file_handle.write("orientation: \n")
                file_handle.write(f"{self.clean_single_float(ori.x)}\n")
                file_handle.write(f"{self.clean_single_float(ori.y)}\n")
                file_handle.write(f"{self.clean_single_float(ori.z)}\n")
                file_handle.write(f"{self.clean_single_float(ori.w)}\n")
            elif hasattr(msg, 'sample_time'):
                # LBRState类型 - 复杂的状态数据
                # 写入所有状态和关节数据
                # data_list = [
                #     msg.sample_time, msg.session_state, msg.connection_quality,
                #     msg.safety_state, msg.operation_mode, msg.drive_state,
                #     msg.client_command_mode, msg.overlay_type, msg.control_mode,
                #     msg.time_stamp_sec, msg.time_stamp_nano_sec
                # ]
                data_list = []
                data_list.extend(msg.measured_joint_position)
                data_list.extend(msg.commanded_joint_position)
                data_list.extend(msg.measured_torque)
                data_list.extend(msg.commanded_torque)
                data_list.extend(msg.external_torque)
                data_list.extend(msg.ipo_joint_position)
                data_list.append(msg.tracking_performance)
                
                for val in data_list:
                    cleaned_val = self.clean_single_float(val)
                    file_handle.write(f"{cleaned_val:.6f}\n")
            elif hasattr(msg, 'position') and hasattr(msg, 'velocity'):
                # JointState类型
                # 处理位置数据
                pos = list(msg.position) if msg.position else []
                vel = list(msg.velocity) if msg.velocity else []
                eff = list(msg.effort) if hasattr(msg, 'effort') and msg.effort else []
                
                # 确保有7个关节的数据
                while len(pos) < 7:
                    pos.append(0.0)
                while len(vel) < 7:
                    vel.append(0.0)
                while len(eff) < 7:
                    eff.append(0.0)
                
                # 写入所有数据
                file_handle.write("joint_pos: \n")
                for val in (pos[:7]):
                    cleaned_val = self.clean_single_float(val)
                    file_handle.write(f"{cleaned_val:.6f}\n")
                file_handle.write("joint_vel: \n")
                for val in (vel[:7]):
                    cleaned_val = self.clean_single_float(val)
                    file_handle.write(f"{cleaned_val:.6f}\n")
                file_handle.write("joint_eff: \n")
                for val in (eff[:7]):
                    cleaned_val = self.clean_single_float(val)
                    file_handle.write(f"{cleaned_val:.6f}\n")
            elif hasattr(msg, 'data'):
                # Float32类型
                cleaned_val = self.clean_single_float(msg.data)
                file_handle.write(f"{cleaned_val:.6f}\n")
            else:
                # 其他未知类型
                file_handle.write("unknown_type\n")
            
            # 写入30个*号作为分隔符
            file_handle.write("*" * 30 + "\n")
            file_handle.flush()
            
        except Exception as e:
            # 出错时写入错误信息
            file_handle = self.data_files[topic_name]
            file_handle.write(f"time: {receive_time}\n")
            file_handle.write("error_data\n")
            file_handle.write("*" * 30 + "\n")
            file_handle.flush()

    def extract_csv_data(self, msg, receive_time):
        """提取消息数据为CSV行数据"""
        try:
            row = [receive_time]
            
            # 根据消息类型提取数据
            if hasattr(msg, 'joint_position') and hasattr(msg, 'wrench'):
                # LBRWrenchCommand类型
                row.extend(msg.joint_position)
                row.extend(msg.wrench)
            elif hasattr(msg, 'sample_time'):
                # LBRState类型
                row.extend([
                    msg.sample_time, msg.session_state, msg.connection_quality,
                    msg.safety_state, msg.operation_mode, msg.drive_state,
                    msg.client_command_mode, msg.overlay_type, msg.control_mode,
                    msg.time_stamp_sec, msg.time_stamp_nano_sec
                ])
                row.extend(msg.measured_joint_position)
                row.extend(msg.commanded_joint_position)
                row.extend(msg.measured_torque)
                row.extend(msg.commanded_torque)
                row.extend(msg.external_torque)
                row.extend(msg.ipo_joint_position)
                row.append(msg.tracking_performance)
            elif hasattr(msg, 'position') and hasattr(msg, 'velocity'):
                # JointState类型
                # 确保有7个关节的数据，不足的用0填充
                pos = list(msg.position) if msg.position else []
                vel = list(msg.velocity) if msg.velocity else []
                eff = list(msg.effort) if hasattr(msg, 'effort') and msg.effort else []
                
                # 填充到7个关节
                while len(pos) < 7:
                    pos.append(0.0)
                while len(vel) < 7:
                    vel.append(0.0)
                while len(eff) < 7:
                    eff.append(0.0)
                
                row.extend(pos[:7])  # 只取前7个
                row.extend(vel[:7])
                row.extend(eff[:7])
            elif hasattr(msg, 'x') and hasattr(msg, 'y') and hasattr(msg, 'z'):
                # Array3类型
                row.extend([msg.x, msg.y, msg.z])
            elif hasattr(msg, 'data'):
                # Float32类型
                row.append(msg.data)
            else:
                # 其他类型，作为字符串处理
                row.append(str(msg)[:200])
                
            return row
        except Exception as e:
            # 出错时返回错误信息
            return [receive_time, f"extract_error: {str(e)}"]
    
    def extract_data(self, msg):
        """提取消息数据为字符串（保留用于兼容性）"""
        try:
            if hasattr(msg, 'x') and hasattr(msg, 'y') and hasattr(msg, 'z'):
                # Array3 类型
                return f"x={msg.x}, y={msg.y}, z={msg.z}"
            elif hasattr(msg, 'row1') and hasattr(msg, 'row2') and hasattr(msg, 'row3'):
                # Cloud 类型
                try:
                    num_points = len(msg.row1) if msg.row1 is not None else 0
                    return f"points={num_points}"
                except:
                    return "points=0"
            elif hasattr(msg, 'data'):
                # Float32 类型
                return str(msg.data)
            elif hasattr(msg, 'position') and hasattr(msg, 'velocity'):
                # JointState 类型
                try:
                    pos_str = ','.join(map(str, msg.position)) if msg.position is not None else ""
                    vel_str = ','.join(map(str, msg.velocity)) if msg.velocity is not None else ""
                    return f"pos=[{pos_str}], vel=[{vel_str}]"
                except:
                    return "pos=[], vel=[]"
            else:
                # 其他类型，返回字符串表示
                return str(msg)[:200]  # 限制长度
        except Exception as e:
            return f"extract_error: {str(e)}"
    
    def write_image_data(self, topic_name, msg, receive_time, header_time):
        """写入Image类型的数据到txt文件 - 从32FC3图像提取20*20*3数据"""
        try:
            file_handle = self.data_files[topic_name]
            
            # 写入时间戳信息
            file_handle.write(f"time: {receive_time/1e9}\n")
            
            try:
                if msg.encoding == "32FC3":
                    # 将Image消息转换为numpy数组
                    force_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC3")
                    # force_image shape: (20, 20, 3)
                    
                    # 提取XYZ数据
                    x_data = force_image[:, :, 0].flatten()  # X通道
                    y_data = force_image[:, :, 1].flatten()  # Y通道  
                    z_data = force_image[:, :, 2].flatten()  # Z通道
                    
                    # 清理数据
                    x_clean = self.clean_float_array(x_data, 400)
                    y_clean = self.clean_float_array(y_data, 400)
                    z_clean = self.clean_float_array(z_data, 400)
                    
                    # 写入XYZ三行数据
                    x_str = ' '.join(f"{value:.6f}" for value in x_clean)
                    file_handle.write(f"{x_str}\n")
                    
                    y_str = ' '.join(f"{value:.6f}" for value in y_clean)
                    file_handle.write(f"{y_str}\n")
                    
                    z_str = ' '.join(f"{value:.6f}" for value in z_clean)
                    file_handle.write(f"{z_str}\n")
                else:
                    # 不支持的编码，写入NaN数据
                    nan_line = ' '.join(["nan"] * 400)
                    file_handle.write(f"{nan_line}\n")
                    file_handle.write(f"{nan_line}\n")
                    file_handle.write(f"{nan_line}\n")
            except Exception as e:
                # 转换失败，写入NaN数据
                nan_line = ' '.join(["nan"] * 400)
                file_handle.write(f"{nan_line}\n")
                file_handle.write(f"{nan_line}\n")
                file_handle.write(f"{nan_line}\n")
            
            # 写入*号分隔符
            file_handle.write("*" * 30 + "\n")
            file_handle.flush()
            
        except Exception as e:
            # 出错时写入错误信息
            file_handle = self.data_files[topic_name]
            file_handle.write(f"time: {receive_time}\n")
            # 写入400个NaN的XYZ数据
            nan_line = ' '.join(["nan"] * 400)
            file_handle.write(f"{nan_line}\n")
            file_handle.write(f"{nan_line}\n")
            file_handle.write(f"{nan_line}\n")
            file_handle.write("*" * 30 + "\n")
            file_handle.flush()

    def write_cloud_data(self, topic_name, msg, receive_time, header_time):
        """写入Cloud类型的数据到txt文件 - 20*20*3点阵，时间戳+XYZ分别换行+*号分隔"""
        # return TypeError("Cloud data writing is deprecated in favor of Image data.")
        try:
            file_handle = self.data_files[topic_name]
            
            # 写入时间戳信息
            file_handle.write(f"time: {receive_time/1e9}\n")
            
            # 处理Cloud消息结构 - 每个row包含400个点的坐标
            x_data = []
            y_data = []
            z_data = []
            
            if hasattr(msg, 'row1') and hasattr(msg, 'row2') and hasattr(msg, 'row3'):
                # 处理X数据 (row1) - 400个X坐标
                if msg.row1 is not None:
                    x_raw = list(msg.row1)[:400]  # 取前400个点
                    x_data = self.clean_float_array(x_raw, 400)
                else:
                    x_data = [-99.0] * 400
                
                # 处理Y数据 (row2) - 400个Y坐标
                if msg.row2 is not None:
                    y_raw = list(msg.row2)[:400]  # 取前400个点
                    y_data = self.clean_float_array(y_raw, 400)
                else:
                    y_data = [-99.0] * 400
                
                # 处理Z数据 (row3) - 400个Z坐标
                if msg.row3 is not None:
                    z_raw = list(msg.row3)[:400]  # 取前400个点
                    z_data = self.clean_float_array(z_raw, 400)
                else:
                    z_data = [-99.0] * 400
            else:
                # 如果没有有效数据，创建三组400个-99
                x_data = [-99.0] * 400
                y_data = [-99.0] * 400
                z_data = [-99.0] * 400
            
            # 写入XYZ三行数据，每行400个点，用空格分隔
            x_str = ' '.join(f"{value:.6f}" for value in x_data)
            file_handle.write("x_str: \n")
            file_handle.write(f"{x_str}\n")
            
            y_str = ' '.join(f"{value:.6f}" for value in y_data)
            file_handle.write("y_str: \n")
            file_handle.write(f"{y_str}\n")
            
            z_str = ' '.join(f"{value:.6f}" for value in z_data)
            file_handle.write("z_str: \n")
            file_handle.write(f"{z_str}\n")
            
            # 写入*号分隔符
            file_handle.write("*" * 30 + "\n")
            file_handle.flush()
            
        except Exception as e:
            # 出错时写入错误信息
            file_handle = self.data_files[topic_name]
            file_handle.write(f"time: {receive_time}\n")
            # 写入400个-99的XYZ数据
            error_line = ' '.join(["-99.000000"] * 400)
            file_handle.write(f"{error_line}\n")
            file_handle.write(f"{error_line}\n") 
            file_handle.write(f"{error_line}\n")
            file_handle.write("*" * 30 + "\n")
            file_handle.flush()
    
    def clean_single_float(self, value):
        """清理单个浮点数，处理NaN和无穷大等奇异值"""
        import math
        
        try:
            # 转换为浮点数
            float_val = float(value)
            
            # 检查是否为NaN或无穷大
            if math.isnan(float_val) or math.isinf(float_val):
                return -99.0
            # 检查是否为异常大的数值（绝对值超过1e10）
            elif abs(float_val) > 1e10:
                return -99.0
            else:
                return float_val
        except (ValueError, TypeError):
            # 转换失败时返回-99
            return -99.0

    def clean_float_array(self, raw_data, target_length):
        """清理浮点数组，处理NaN和无穷大等奇异值"""
        import math
        
        cleaned_data = []
        for value in raw_data:
            try:
                # 转换为浮点数
                float_val = float(value)
                
                # 检查是否为NaN或无穷大
                if math.isnan(float_val) or math.isinf(float_val):
                    cleaned_data.append(-99.0)
                # 检查是否为异常大的数值（绝对值超过1e10）
                elif abs(float_val) > 1e10:
                    cleaned_data.append(-99.0)
                else:
                    cleaned_data.append(float_val)
            except (ValueError, TypeError):
                # 转换失败时用-99填充
                cleaned_data.append(-99.0)
        
        # 确保数组长度正确
        while len(cleaned_data) < target_length:
            cleaned_data.append(-99.0)
        
        return cleaned_data[:target_length]
    
    def generate_summary(self):
        """生成数据采集摘要"""
        summary_file = os.path.join(self.unique_dir, 'summary.txt')
        with open(summary_file, 'w') as f:
            f.write("数据采集摘要\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"采集时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据目录: {self.unique_dir}\n")
            f.write(f"监控话题数: {len(self.topic_types)}\n\n")
            f.write(f"话题说明： \n")
            f.write(f"lbr_state_pose：机械臂末端（不包括夹爪）在极坐标系中的位姿。 \n")
            f.write(f"_lbr_joint_states：机械臂各个关节的关节角度、角速度和关节力矩。 \n")
            f.write(f"positions_r/l: 右/左触觉传感器上标志点在传感器坐标系中的位置，具体见传感器文档，保存时20*20*3点阵按照XYZ方向分开保存成3个400*1数组数据。 \n")
            f.write(f"displancements_r/l: 右/左触觉传感器上标志点在传感器坐标系中的位移，具体见传感器文档，保存时20*20*3点阵按照XYZ方向分开保存成3个400*1数组数据。 \n")
            f.write(f"forces_r/l: 右/左触觉传感器上标志点处所受的力矢量，具体见传感器文档，保存时20*20*3点阵按照XYZ方向分开保存成3个400*1数组数据。 \n")
            f.write(f"resultant_force_r/l: 右/左触觉传感器上所受的合力，具体见传感器文档。 \n")
            f.write(f"resultant_moment_r/l: 右/左触觉传感器上所受的合力矩，具体见传感器文档。 \n")
            f.write(f"index_r/l: 右/左触觉传感器的编号项，具体见传感器文档。 \n\n")
            f.write(f"- 所有非法数据(NaN, 无穷大, 绝对值>1e10)记录为-99\n")
        
        self.get_logger().info(f"摘要已生成: {summary_file}")

    def destroy_node(self):
        """重写销毁节点方法，确保正确清理订阅"""
        try:
            # 清理订阅
            if hasattr(self, '_subscriptions'):
                for subscription in self._subscriptions:
                    try:
                        self.destroy_subscription(subscription)
                    except Exception as e:
                        self.get_logger().warn(f"清理订阅时出错: {str(e)}")
                self._subscriptions.clear()
            
            # 关闭文件
            for file_handle in self.data_files.values():
                if not file_handle.closed:
                    file_handle.close()
        except Exception as e:
            if hasattr(self, 'get_logger'):
                self.get_logger().error(f"销毁节点时出错: {str(e)}")
        finally:
            # 调用父类的销毁方法
            super().destroy_node()
    
    def __del__(self):
        """析构函数，确保文件正确关闭"""
        try:
            for file_handle in self.data_files.values():
                if not file_handle.closed:
                    file_handle.close()
        except Exception as e:
            if hasattr(self, 'get_logger'):
                self.get_logger().error(f"关闭文件时出错: {str(e)}")


def main(args=None):
    rclpy.init(args=args)
    
    print("启动数据记录器...")
    print("功能: 记录各个话题的数据和到达时间戳")
    print()
    
    recorder = None
    try:
        recorder = DatasetRecorder()
        print(f"数据将保存到: {recorder.unique_dir}")
        print("按 Ctrl+C 停止记录")
        print()
        
        rclpy.spin(recorder)
    except KeyboardInterrupt:
        print("\n正在停止数据记录...")
    except Exception as e:
        print(f"启动失败: {e}")
    finally:
        if recorder is not None:
            print("正在生成摘要...")
            recorder.generate_summary()
            print(f"所有数据已保存到: {recorder.unique_dir}")
            try:
                recorder.destroy_node()
            except Exception as e:
                print(f"销毁节点时出错: {e}")
        
        try:
            rclpy.shutdown()
        except Exception as e:
            print(f"关闭ROS时出错: {e}")


if __name__ == '__main__':
    main()
