# ros2 topic echo /lbr/state/pose --once
# ros2 topic pub --once /lbr/command/pose geometry_msgs/msg/Pose \
# "{
#   position: {x: 0.45, y: 0.0, z: 0.25},
#   orientation: {x: 0.0, y: 1.0, z: 0.0, w: 0.0}
# }"
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from lbr_fri_idl.msg import LBRState
from lbr_fri_idl.msg import LBRJointPositionCommand
import numpy as np
import time

class LBRController(Node):
    def __init__(self):
        super().__init__('initial_controller')
        self.subscription = self.create_subscription(
            LBRState,
            '/lbr/state',
            self.current_joint_position_callback,
            10)
        self.publisher = self.create_publisher(
            LBRJointPositionCommand,
            '/lbr/command/joint_position',
            QoSProfile(depth=10))
        self.current_joint_position = None
        self.target_joint_position = [0.0, 0.349, 0.0, -1.222, 0.0, 1.571, 0.0]
        # - Translation: [0.544, -0.000, 0.629]
        # - Rotation: in Quaternion (xyzw) [-0.000, 1.000, -0.000, -0.000]
        # - Rotation: in RPY (radian) [-3.142, -0.000, -3.142]
        # - Rotation: in RPY (degree) [-180.000, -0.023, -180.000]
        self.rate = 100  # 降低发布速率为10Hz，可以根据具体情况调整

    def current_joint_position_callback(self, msg):
        self.current_joint_position = msg.measured_joint_position
        self.get_logger().info(f'Received positions: {self.current_joint_position}')
        self.send_joint_positions()

    def interpolate_joint_positions(self, start_positions, end_positions, duration):
        num_steps = int(duration * self.rate)
        trajectory = []

        for i in range(num_steps + 1):
            alpha = i / num_steps
            interpolated_positions = [
                start + alpha * (end - start)
                for start, end in zip(start_positions, end_positions)
            ]
            trajectory.append(interpolated_positions)

        return trajectory


    def send_joint_positions(self):
        if self.current_joint_position is not None:
            trajectory = self.interpolate_joint_positions(self.current_joint_position,
                                                          self.target_joint_position,
                                                          duration=10.0)
            for positions in trajectory:
                msg = LBRJointPositionCommand()
                msg.joint_position = positions
                self.publisher.publish(msg)
                self.get_logger().info(f'Published positions: {msg.joint_position}')
                # rclpy.spin_once(self)  # 处理任何回调以更新当前关节位置
                time.sleep(1.0 / self.rate)
         # 发送完毕后销毁节点
            self.get_logger().info('Finished sending joint positions. Shutting down...')
            self.destroy_node()
        else:
            self.get_logger().warn('Waiting for current joint positions...')

def main(args=None):
    rclpy.init(args=args)
    initial_controller = LBRController()
    rclpy.spin(initial_controller)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
