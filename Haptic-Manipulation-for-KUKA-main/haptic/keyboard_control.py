"""
键盘控制节点 - 通过键盘控制机器人笛卡尔空间运动
基于cartesian_controller和down.sh逻辑
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Quaternion
import sys
import termios
import tty
import threading


class KeyboardControlNode(Node):
    """键盘控制节点"""
    
    def __init__(self):
        super().__init__('keyboard_control_node')
        
        # 当前位姿
        self.current_pose = None
        
        # 移动步长 (米)
        self.declare_parameter('step_size', 0.01)  # 默认1cm
        self.step_size = self.get_parameter('step_size').value
        
        # 锁定的目标姿态 (x=0, y=1, z=0, w=0)
        self.target_orientation = Quaternion(x=0.0, y=1.0, z=0.0, w=0.0)
        
        # 订阅当前位姿
        self.pose_sub = self.create_subscription(
            Pose, '/lbr/state/pose', self.pose_callback, 10)
        
        # 发布到/ab_action（会由cartesian_controller插值）
        self.pose_pub = self.create_publisher(Pose, '/ab_action', 10)
        
        # 键盘输入线程锁
        self.input_lock = threading.Lock()
        
        self.get_logger().info("键盘控制节点启动")
        self.get_logger().info(f"移动步长: {self.step_size*1000:.1f}mm")
        self.print_instructions()
    
    def pose_callback(self, msg):
        """当前位姿回调"""
        self.current_pose = msg
    
    def print_instructions(self):
        """打印控制说明"""
        print("\n" + "="*50)
        print("键盘控制说明:")
        print("="*50)
        print("  W/S: X方向前进/后退")
        print("  A/D: Y方向左移/右移")
        print("  Q/E: Z方向上升/下降")
        print("  +/-: 增加/减小步长")
        print("  R:   复位到原点")
        print("  SPACE: 显示当前位姿")
        print("  ESC/Ctrl+C: 退出")
        print("="*50)
        print(f"当前步长: {self.step_size*1000:.1f}mm\n")
    
    def get_key(self):
        """获取键盘输入（非阻塞）"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            key = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return key
    
    def move_delta(self, dx, dy, dz):
        """发送增量移动命令"""
        if self.current_pose is None:
            self.get_logger().warn("等待位姿初始化...")
            return
        
        # 计算目标位姿
        target_pose = Pose()
        target_pose.position.x = self.current_pose.position.x + dx
        target_pose.position.y = self.current_pose.position.y + dy
        target_pose.position.z = self.current_pose.position.z + dz
        target_pose.orientation = self.target_orientation
        
        # 发布到/ab_action
        self.pose_pub.publish(target_pose)
        
        self.get_logger().info(
            f"移动命令: Δ[{dx:+.4f}, {dy:+.4f}, {dz:+.4f}] -> "
            f"目标位姿: [{target_pose.position.x:.4f}, {target_pose.position.y:.4f}, {target_pose.position.z:.4f}]")
    
    def reset_to_home(self):
        """复位到原点 (0.6, 0.15, 0.3)"""
        home_pose = Pose()
        home_pose.position.x = 0.6
        home_pose.position.y = 0.15
        home_pose.position.z = 0.3
        home_pose.orientation = self.target_orientation
        
        self.pose_pub.publish(home_pose)
        self.get_logger().info("复位到原点: [0.6, 0.15, 0.3]")
    
    def show_current_pose(self):
        """显示当前位姿"""
        if self.current_pose is None:
            print("位姿尚未初始化")
            return
        
        print(f"\n当前位姿:")
        print(f"  X: {self.current_pose.position.x:.6f} m")
        print(f"  Y: {self.current_pose.position.y:.6f} m")
        print(f"  Z: {self.current_pose.position.z:.6f} m")
        print(f"当前步长: {self.step_size*1000:.1f}mm\n")
    
    def adjust_step_size(self, factor):
        """调整步长"""
        self.step_size *= factor
        # 限制步长范围 [0.001m, 0.1m] = [1mm, 10cm]
        self.step_size = max(0.001, min(0.1, self.step_size))
        print(f"步长调整为: {self.step_size*1000:.1f}mm")
    
    def run(self):
        """运行键盘控制循环"""
        print("等待位姿初始化...")
        
        # 等待位姿初始化
        while self.current_pose is None and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
        
        print("位姿已初始化，可以开始控制！\n")
        
        try:
            while rclpy.ok():
                # 处理ROS事件
                rclpy.spin_once(self, timeout_sec=0.001)
                
                # 读取键盘输入
                key = self.get_key()
                
                # 处理按键
                if key == '\x1b':  # ESC
                    print("\n退出键盘控制")
                    break
                elif key == '\x03':  # Ctrl+C
                    print("\n退出键盘控制")
                    break
                elif key.lower() == 'w':
                    self.move_delta(self.step_size, 0, 0)  # X+
                elif key.lower() == 's':
                    self.move_delta(-self.step_size, 0, 0)  # X-
                elif key.lower() == 'a':
                    self.move_delta(0, self.step_size, 0)  # Y+
                elif key.lower() == 'd':
                    self.move_delta(0, -self.step_size, 0)  # Y-
                elif key.lower() == 'q':
                    self.move_delta(0, 0, self.step_size)  # Z+
                elif key.lower() == 'e':
                    self.move_delta(0, 0, -self.step_size)  # Z-
                elif key == '+' or key == '=':
                    self.adjust_step_size(1.5)  # 增加步长50%
                elif key == '-' or key == '_':
                    self.adjust_step_size(0.667)  # 减小步长33%
                elif key.lower() == 'r':
                    self.reset_to_home()
                elif key == ' ':
                    self.show_current_pose()
                elif key.lower() == 'h':
                    self.print_instructions()
        
        except Exception as e:
            self.get_logger().error(f"控制循环错误: {e}")
    
    def destroy_node(self):
        """销毁节点"""
        self.get_logger().info("正在停止键盘控制节点...")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    print("\n" + "="*50)
    print("  KUKA 键盘控制节点")
    print("="*50)
    print("订阅: /lbr/state/pose (Pose) - 当前位姿")
    print("发布: /ab_action (Pose) - 目标位姿（会被cartesian_controller插值）")
    print("="*50 + "\n")
    
    node = None
    try:
        node = KeyboardControlNode()
        node.run()
        
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
