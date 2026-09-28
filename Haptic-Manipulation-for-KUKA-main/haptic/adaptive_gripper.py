"""ROS 2 constant-force controller for a Changingtek gripper."""

import math
import signal
import threading
import time

import rclpy
from geometry_msgs.msg import Vector3
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import Bool, String

from .Changingtek_rtu_psdk import Changingtek_rtu_psdk, find_valid_port


class PIDController:
    def __init__(self, k_p, k_i, k_d):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.previous_error = 0.0
        self.integral_error = 0.0

    def update(self, error):
        self.integral_error += error
        derivative_error = error - self.previous_error
        self.previous_error = error
        return -(
            self.k_p * error
            + self.k_i * self.integral_error
            + self.k_d * derivative_error
        )


class AdaptiveGripper(Node):
    active_instance = None
    shutdown_requested = None

    def __init__(self):
        super().__init__('adaptive_gripper')
        AdaptiveGripper.active_instance = self
        self.gripper = None
        self.finished = False
        self.hardware_stopped = False
        self.shutdown_hold_complete = False
        self.command_lock = threading.Lock()
        self.last_force_time = None
        self.target_position = 0.0
        self.current_position = 0
        self.declare_parameter('port', '')
        self.declare_parameter('slave_id', 1)
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('force_topic', '/resultant_force_r')
        self.declare_parameter('target_force', 9.5)
        self.declare_parameter('force_tolerance', 0.5)
        self.declare_parameter('stable_frames', 50)
        self.declare_parameter('force_timeout', 0.5)
        self.declare_parameter('command_interval', 0.1)
        self.declare_parameter('feedback_interval', 0.2)
        self.declare_parameter('close_limit_mm', 120.0)
        self.declare_parameter('speed', 15)
        self.declare_parameter('torque_limit', 90)
        self.declare_parameter('kp', 0.5)
        self.declare_parameter('ki', 0.0)
        self.declare_parameter('kd', 0.5)

        self.target_force = float(self.get_parameter('target_force').value)
        self.force_tolerance = float(self.get_parameter('force_tolerance').value)
        self.stable_frames_required = int(self.get_parameter('stable_frames').value)
        self.force_timeout = float(self.get_parameter('force_timeout').value)
        self.command_interval = float(
            self.get_parameter('command_interval').value)
        self.feedback_interval = float(
            self.get_parameter('feedback_interval').value)
        self.close_limit = round(float(self.get_parameter('close_limit_mm').value) * 100)
        self.speed = int(self.get_parameter('speed').value)
        self.torque_limit = int(self.get_parameter('torque_limit').value)
        self._validate_parameters()

        notification_qos = QoSProfile(depth=1)
        notification_qos.reliability = ReliabilityPolicy.RELIABLE
        notification_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.complete_publisher = self.create_publisher(
            Bool, '~/grasp_complete', notification_qos)
        self.status_publisher = self.create_publisher(
            String, '~/status', notification_qos)
        self.complete_publisher.publish(Bool(data=False))
        self._publish_status('connecting')

        slave_id = int(self.get_parameter('slave_id').value)
        baudrate = int(self.get_parameter('baudrate').value)
        configured_port = str(self.get_parameter('port').value)
        port = configured_port or find_valid_port(slave_id, baudrate)
        if not port:
            raise RuntimeError(
                'No responding Changingtek gripper found; set the port parameter')

        self.gripper = Changingtek_rtu_psdk(
            port, slave_id, baudrate, timeout=1.0)
        self.gripper.instrument.clear_buffers_before_each_transaction = True
        if not self.gripper.connect():
            raise RuntimeError(f'Failed to connect to gripper on {port}')
        self.gripper.enable(True)
        self.gripper.set_cmd_update_mode(0)
        self.gripper.temp_move(0, self.speed, self.torque_limit, 100, 100, True)
        open_result = self._wait_until_pos_or_torque_interruptible(10.0)
        if open_result == 'shutdown':
            self.stop_for_shutdown()
            raise KeyboardInterrupt
        if open_result != 'position':
            raise RuntimeError(
                f'Could not confirm fully open position (result={open_result})')

        self.pid = PIDController(
            float(self.get_parameter('kp').value),
            float(self.get_parameter('ki').value),
            float(self.get_parameter('kd').value),
        )
        self.stable_frames = 0
        self.last_commanded_position = 0
        self.last_command_time = 0.0
        self.last_feedback_time = 0.0
        force_topic = str(self.get_parameter('force_topic').value)
        self.force_subscription = self.create_subscription(
            Vector3, force_topic, self._force_callback, 10)
        self.force_watchdog = self.create_timer(0.1, self._check_force_timeout)
        self.get_logger().info(
            f'Gripper connected on {port}; waiting for force data on {force_topic}')
        self._publish_status('closing')

    def _validate_parameters(self):
        if self.target_force <= 0.0:
            raise ValueError('target_force must be positive')
        if self.force_tolerance < 0.0:
            raise ValueError('force_tolerance must be non-negative')
        if self.stable_frames_required <= 0:
            raise ValueError('stable_frames must be positive')
        if self.force_timeout <= 0.0:
            raise ValueError('force_timeout must be positive')
        if self.command_interval <= 0.0:
            raise ValueError('command_interval must be positive')
        if self.feedback_interval <= 0.0:
            raise ValueError('feedback_interval must be positive')
        if self.close_limit <= 0:
            raise ValueError('close_limit_mm must be positive')
        if not 0 <= self.speed <= 100:
            raise ValueError('speed must be between 0 and 100')
        if not 0 <= self.torque_limit <= 100:
            raise ValueError('torque_limit must be between 0 and 100')

    def _force_callback(self, message):
        if self.finished or self._shutdown_is_requested():
            return

        self.last_force_time = time.monotonic()
        force = math.sqrt(message.z * message.z)
        error = self.target_force - force

        with self.command_lock:
            if self._shutdown_is_requested():
                return
            if abs(error) <= self.force_tolerance:
                self.stable_frames += 1
            else:
                self.stable_frames = 0

            if self.stable_frames >= self.stable_frames_required:
                self.finished = True
                held_position = self._hold_current_position()
                self.complete_publisher.publish(Bool(data=True))
                self._publish_status('complete')
                self.get_logger().info(
                    'Grasp complete: force remained within tolerance for '
                    f'{self.stable_frames_required} frames; holding at '
                    f'{held_position / 100:.2f} mm')
                return

            if self.gripper.torque_reached():
                self.finished = True
                held_position = self._safe_stop_and_hold()
                self._publish_status('torque_limit_reached')
                self.get_logger().error(
                    'Gripper torque limit reached before force stabilized; '
                    f'holding at {held_position / 100:.2f} mm')
                return

            self.target_position -= self.pid.update(error)
            self.target_position = min(
                max(self.target_position, 0.0), float(self.close_limit))
            if self._shutdown_is_requested():
                return
            self._send_position_if_needed(round(self.target_position))
            now = time.monotonic()
            if now - self.last_feedback_time >= self.feedback_interval:
                self.current_position = self.gripper.feedback_position()
                self.last_feedback_time = now
                self.get_logger().info(
                    f'force={force:.3f} N, current/target position='
                    f'{self.current_position / 100:.2f}/'
                    f'{self.target_position / 100:.2f} mm, '
                    f'stable_frames={self.stable_frames}/'
                    f'{self.stable_frames_required}')

    def _send_position_if_needed(self, commanded_position):
        now = time.monotonic()
        if commanded_position == self.last_commanded_position:
            return
        if now - self.last_command_time < self.command_interval:
            return

        self.gripper.set_temp_position_mm(commanded_position)
        self.gripper.trigger_temp_move()
        self.last_commanded_position = commanded_position
        self.last_command_time = now

    def _check_force_timeout(self):
        if self.finished or self.last_force_time is None:
            return
        if time.monotonic() - self.last_force_time <= self.force_timeout:
            return

        with self.command_lock:
            if self.finished:
                return
            self.finished = True
            held_position = self._safe_stop_and_hold()
            self._publish_status('force_data_timeout')
            self.get_logger().error(
                f'No force data for {self.force_timeout:.2f} s; '
                f'holding at {held_position / 100:.2f} mm')

    def _hold_current_position(self):
        gripper = self.gripper
        if gripper is None:
            return None
        held_position = self._retry_serial(
            gripper.feedback_position, 'read current position')
        self._retry_serial(
            lambda: gripper.temp_move(
                held_position, self.speed,
                self.torque_limit, 100, 100, True),
            'command position hold')
        self.target_position = float(held_position)
        self.current_position = held_position
        return held_position

    def _safe_stop_and_hold(self):
        gripper = self.gripper
        if gripper is None:
            return None

        self.finished = True
        self._retry_serial(lambda: gripper.enable(False), 'disable gripper')
        self.hardware_stopped = True
        time.sleep(0.05)

        try:
            held_position = self._retry_serial(
                gripper.feedback_position, 'read stopped position')
        except Exception:
            held_position = int(self.current_position)

        try:
            self._retry_serial(
                lambda: gripper.temp_move(
                    held_position, self.speed,
                    self.torque_limit, 100, 100, True),
                'program stopped position')
            self._retry_serial(lambda: gripper.enable(True), 're-enable gripper')
            self.hardware_stopped = False
        except Exception:
            try:
                self._retry_serial(
                    lambda: gripper.enable(False), 'keep gripper disabled')
            finally:
                self.hardware_stopped = True
            raise

        self.target_position = float(held_position)
        self.current_position = held_position
        return held_position

    def _retry_serial(self, operation, description, attempts=3):
        last_error = None
        for attempt in range(1, attempts + 1):
            try:
                serial_port = self.gripper.instrument.serial
                serial_port.reset_input_buffer()
                return operation()
            except Exception as error:
                last_error = error
                if attempt < attempts:
                    time.sleep(0.05)
        raise RuntimeError(
            f'{description} failed after {attempts} attempts: {last_error}') \
            from last_error

    def _wait_until_pos_or_torque_interruptible(self, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._shutdown_is_requested():
                return 'shutdown'
            try:
                if self.gripper.position_reached():
                    return 'position'
                if self.gripper.torque_reached():
                    return 'torque'
            except Exception:
                pass
            time.sleep(0.02)
        return 'timeout'

    def _shutdown_is_requested(self):
        event = AdaptiveGripper.shutdown_requested
        return event is not None and event.is_set()

    def stop_for_shutdown(self):
        gripper = getattr(self, 'gripper', None)
        if gripper is None or self.shutdown_hold_complete:
            return
        with self.command_lock:
            try:
                held_position = self._safe_stop_and_hold()
                self.shutdown_hold_complete = True
                self.get_logger().info(
                    f'Ctrl+C received; holding gripper at '
                    f'{held_position / 100:.2f} mm')
            except Exception as error:
                if self.hardware_stopped:
                    self.get_logger().error(
                        'Position hold failed; gripper remains disabled for '
                        f'safety: {error}')
                else:
                    self.get_logger().error(
                        f'Unable to confirm safe stop: {error}')

    def _publish_status(self, status):
        self.status_publisher.publish(String(data=status))

    def destroy_node(self):
        gripper = getattr(self, 'gripper', None)
        if gripper is not None:
            gripper.disconnect()
            self.gripper = None
        AdaptiveGripper.active_instance = None
        return super().destroy_node()


def main(args=None):
    shutdown_event = threading.Event()
    AdaptiveGripper.shutdown_requested = shutdown_event

    def request_shutdown(signum, frame):
        del signum, frame
        shutdown_event.set()

    previous_sigint = signal.signal(signal.SIGINT, request_shutdown)
    previous_sigterm = signal.signal(signal.SIGTERM, request_shutdown)
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = None
    try:
        node = AdaptiveGripper()
        while rclpy.ok() and not shutdown_event.is_set():
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        if node is not None:
            node.get_logger().fatal(str(error))
        else:
            print(f'adaptive_gripper: {error}')
        raise
    finally:
        active_node = node or AdaptiveGripper.active_instance
        if active_node is not None:
            active_node.stop_for_shutdown()
            active_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        AdaptiveGripper.shutdown_requested = None
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == '__main__':
    main()
