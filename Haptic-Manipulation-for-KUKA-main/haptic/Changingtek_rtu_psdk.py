
"""
Changingtek_rtu_psdk (Modbus RTU over RS-485) -- 平动手，协作手通用
-------------------------------------
这是一个基于 minimalmodbus 的轻量级 Python SDK，用于控制执行器。
提供清晰的读写辅助函数、线程安全支持，以及针对“临时区域”运动和多点运动的便捷方法。

依赖:
    pip install minimalmodbus pyserial

作者: 知行机器人
"""

import time
import threading
import minimalmodbus
import serial
import serial.tools.list_ports

# -----------------------------
# 寄存器映射 (保持寄存器, 0x03/0x06/0x10)
# -----------------------------
# 使能 / 失能
REG_ENABLE                = 0x0100

# 临时区域运动 (写入)
REG_TMP_POS_H             = 0x0102  # 高 16 位
REG_TMP_POS_L             = 0x0103  # 低 16 位
REG_TMP_SPEED             = 0x0104  # 0~100 (占最大速度的百分比 @ 0x0305)
REG_TMP_FORCE             = 0x0105  # 0~100 (占最大力矩的百分比 @ 0x0306)
REG_TMP_ACCEL             = 0x0106
REG_TMP_DECEL             = 0x0107
REG_TMP_TRIGGER           = 0x0108  # 0:空闲, 1:触发

# 多点运动 (写入)
REG_CMD_UPDATE_MODE       = 0x010F  # 0: 立即更新, 1: 忽略更新直到运动结束
REG_MULTI_MODE            = 0x0110  # 0: 顺序, 1: 循环, 2: 选择
REG_MULTI_START_SEG       = 0x0111
REG_MULTI_END_SEG         = 0x0112
REG_MULTI_RESUME_POLICY   = 0x0113  # 0: 继续剩余, 1: 从头开始
REG_MULTI_LOOP_COUNT      = 0x0114  # 0xFFFF 表示无限循环
REG_MULTI_SELECT_SEG      = 0x0116  # 当 REG_MULTI_MODE == 2 时有效
REG_MULTI_TRIGGER         = 0x0117  # 0:空闲, 1:触发
REG_MULTI_PAUSE           = 0x0118  # 0:空闲, 1:暂停

# -----------------------------
# 状态 / 反馈 (只读, 通过 0x03)
# -----------------------------
REG_TORQUE_REACHED        = 0x0601  # 0/1
REG_POS_REACHED           = 0x0602  # 0/1
REG_SPEED_MAX_REACHED     = 0x0603  # 0/1
REG_READY                 = 0x0604  # 0/1 (力矩或位置到达)
REG_CURR_LOOP_COUNT       = 0x0606  # 多点运动中的当前循环计数
REG_CURR_SEG              = 0x0607  # 当前运行的段
REG_POS_FB_H              = 0x060D  # 32位位置反馈的高16位
REG_POS_FB_L              = 0x060E  # 32位位置反馈的低16位
REG_SPEED_FB              = 0x060B
REG_CURRENT_FB            = 0x060C  # 力矩电流反馈
REG_ALARM                 = 0x0612  # 位掩码: 0x01 过温, 0x02 堵转, 0x04 超速, 0x08 初始化故障, 0x10 限位, 0x20 掉电
REG_PARAM_CHANGED         = 0x0614  # 0/1: 存在未保存的参数

class Changingtek_rtu_psdk:
    def __init__(self, port: str, slave_id: int = 1, baudrate: int = 115200, timeout: float = 0.3):
        """创建一个 Modbus RTU 仪表实例。

        参数:
            port: 串口号, 例如 Windows 上的 'COM4' 或 Linux 上的 '/dev/ttyUSB0'。
            slave_id: Modbus 从站地址 (默认为 1)。
            baudrate: 串口波特率, 默认为 115200。
            timeout: 读写超时时间 (秒)。
        """
        self.instrument = minimalmodbus.Instrument(port, slave_id)
        self.instrument.serial.baudrate = baudrate
        self.instrument.serial.bytesize = 8
        self.instrument.serial.parity   = serial.PARITY_NONE
        self.instrument.serial.stopbits = 1
        self.instrument.serial.timeout  = timeout
        self.instrument.mode = minimalmodbus.MODE_RTU
        # 优化: 为了减少延迟，不清除缓冲区 (假设线路干净)
        self.instrument.clear_buffers_before_each_transaction = False
        self._lock = threading.Lock()

    def connect(self) -> bool:
        """连接到串口。"""
        try:
            if not self.instrument.serial.is_open:
                self.instrument.serial.open()
            return True
        except Exception as e:
            print(f"连接错误: {e}")
            return False

    def disconnect(self):
        """关闭串口。"""
        if self.instrument.serial.is_open:
            self.instrument.serial.close()

    # ------------- 低级辅助函数 -------------
    def _w1(self, addr: int, value: int):
        """写单个 16 位保持寄存器 (功能码 0x06)。"""
        with self._lock:
            return self.instrument.write_register(addr, value, functioncode=6)

    def _wn(self, addr: int, values):
        """从 addr 开始写多个 16 位保持寄存器 (功能码 0x10)。"""
        with self._lock:
            return self.instrument.write_registers(addr, list(values))

    def _r(self, addr: int, count: int = 1):
        """读一个或多个 16 位保持寄存器 (功能码 0x03)。"""
        with self._lock:
            return self.instrument.read_registers(addr, count, functioncode=3)

    # ------------- 使能 / 失能 -------------
    def enable(self, enable: bool = True):
        """使能或失能执行器。"""
        return self._w1(REG_ENABLE, 1 if enable else 0)

    # ------------- 临时区域运动 -------------
    def set_temp_position_mm(self, position_mm: int):
        """通过高/低寄存器设置临时目标位置 (32位)。
        position_mm: 整数位置，单位为 mm (根据文档定义的设备单位)。
        """
        if position_mm < 0 or position_mm > 0xFFFFFFFF:
            raise ValueError("position_mm 必须在 0..0xFFFFFFFF 范围内")
        hi = (position_mm >> 16) & 0xFFFF
        lo = position_mm & 0xFFFF
        # 通过功能码 0x10 写两个寄存器
        return self._wn(REG_TMP_POS_H, [hi, lo])

    def set_temp_speed_pct(self, speed_pct: int):
        """设置临时速度，为最大速度的百分比 [0..100]。"""
        if not (0 <= speed_pct <= 100):
            raise ValueError("speed_pct 必须是 0..100")
        return self._w1(REG_TMP_SPEED, speed_pct)

    def set_temp_force_pct(self, force_pct: int):
        """设置临时力矩/力，为最大值的百分比 [0..100]。"""
        if not (0 <= force_pct <= 100):
            raise ValueError("force_pct 必须是 0..100")
        return self._w1(REG_TMP_FORCE, force_pct)

    def set_temp_accel(self, accel: int):
        return self._w1(REG_TMP_ACCEL, accel)

    def set_temp_decel(self, decel: int):
        return self._w1(REG_TMP_DECEL, decel)

    def trigger_temp_move(self):
        """基于临时区域寄存器触发运动。"""
        return self._w1(REG_TMP_TRIGGER, 1)

    def temp_move(self, position_mm: int, speed_pct: int = 100, force_pct: int = 60, accel: int = 2000, decel: int = 2000, trigger: bool = True):
        """便捷方法: 设置所有临时区域参数并 (可选) 触发。"""
        self.set_temp_position_mm(position_mm)
        self.set_temp_speed_pct(speed_pct)
        self.set_temp_force_pct(force_pct)
        self.set_temp_accel(accel)
        self.set_temp_decel(decel)
        if trigger:
            self.trigger_temp_move()

    # ------------- 多点运动 -------------
    def set_cmd_update_mode(self, mode: int):
        """0: 立即更新; 1: 忽略更新直到运动结束。"""
        if mode not in (0, 1):
            raise ValueError("mode 必须是 0 或 1")
        return self._w1(REG_CMD_UPDATE_MODE, mode)

    def set_multi_mode(self, mode: int):
        """0: 顺序, 1: 循环, 2: 选择。"""
        if mode not in (0, 1, 2):
            raise ValueError("mode 必须是 0, 1 或 2")
        return self._w1(REG_MULTI_MODE, mode)

    def set_multi_range(self, start_seg: int, end_seg: int):
        self._w1(REG_MULTI_START_SEG, start_seg)
        return self._w1(REG_MULTI_END_SEG, end_seg)

    def set_multi_resume_policy(self, policy: int):
        """0: 继续剩余; 1: 从头开始。"""
        if policy not in (0, 1):
            raise ValueError("policy 必须是 0 或 1")
        return self._w1(REG_MULTI_RESUME_POLICY, policy)

    def set_multi_loop_count(self, count: int):
        """设置循环计数; 0xFFFF 表示无限循环。"""
        if not (0 <= count <= 0xFFFF):
            raise ValueError("count 必须是 0..0xFFFF")
        return self._w1(REG_MULTI_LOOP_COUNT, count)

    def set_multi_select_segment(self, seg: int):
        return self._w1(REG_MULTI_SELECT_SEG, seg)

    def trigger_multi(self):
        return self._w1(REG_MULTI_TRIGGER, 1)

    def pause_multi(self):
        return self._w1(REG_MULTI_PAUSE, 1)

    # ------------- 状态 / 反馈 -------------
    def _read_bool(self, addr: int) -> bool:
        """内部辅助：读取布尔值状态。"""
        return bool(self._r(addr, 1)[0])

    def torque_reached(self) -> bool:
        """检查是否达到目标力矩（通常用于判断夹取是否成功）。"""
        return self._read_bool(REG_TORQUE_REACHED)

    def position_reached(self) -> bool:
        """检查是否到达目标位置。"""
        return self._read_bool(REG_POS_REACHED)

    def speed_max_reached(self) -> bool:
        """检查是否达到最大速度限制。"""
        return self._read_bool(REG_SPEED_MAX_REACHED)

    def ready(self) -> bool:
        """检查执行器是否就绪（动作完成，位置或力矩已到达）。"""
        return self._read_bool(REG_READY)

    def current_loop_count(self) -> int:
        """读取多点运动模式下的当前循环计数。"""
        return self._r(REG_CURR_LOOP_COUNT, 1)[0]

    def current_segment(self) -> int:
        """读取多点运动模式下的当前运行段号。"""
        return self._r(REG_CURR_SEG, 1)[0]

    def feedback_position(self) -> int:
        """读取当前实时位置。
        返回: 整数位置值 (单位与设置时一致，如 0.01mm)。
        """
        hi, lo = self._r(REG_POS_FB_H, 2)
        return ((hi & 0xFFFF) << 16) | (lo & 0xFFFF)

    def feedback_speed(self) -> int:
        """读取当前实时速度。"""
        return self._r(REG_SPEED_FB, 1)[0]

    def feedback_current(self) -> int:
        """读取当前实时力矩/电流 (百分比或原始值，视配置而定)。"""
        return self._r(REG_CURRENT_FB, 1)[0]

    def read_alarm(self) -> int:
        """读取报警状态码。
        返回: 报警位掩码 (见 REG_ALARM 定义)。
        """
        return self._r(REG_ALARM, 1)[0]

    def param_changed(self) -> bool:
        """检查是否有参数被修改但尚未保存到 EEPROM。"""
        return self._read_bool(REG_PARAM_CHANGED)

    # ------------- 工具函数 -------------
    def wait_until_ready(self, timeout: float = 5.0, poll: float = 0.02) -> bool:
        """轮询 REG_READY 直到为 True 或超时。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if self.ready():
                    return True
            except (minimalmodbus.NoResponseError, serial.SerialException):
                # 瞬时错误: 忽略并在超时内重试
                pass
            time.sleep(poll)
        return False

    def wait_until_pos_or_torque(self, timeout: float = 5.0, poll: float = 0.02) -> str:
        """等待直到位置到达或力矩到达，或超时。
        返回: 'position', 'torque', 或 'timeout'。
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                # 优化: 在一次事务中读取两个状态标志
                # REG_TORQUE_REACHED (0x0601) 和 REG_POS_REACHED (0x0602) 是相邻的
                vals = self._r(REG_TORQUE_REACHED, 2)
                torque_reached = bool(vals[0])
                pos_reached = bool(vals[1])
                
                if pos_reached:
                    return 'position'
                if torque_reached:
                    return 'torque'
            except (minimalmodbus.NoResponseError, serial.SerialException):
                pass
            time.sleep(poll)
        return 'timeout'

def list_ports():
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("未发现串口设备。")
        return
    print("可用串口:")
    for i, p in enumerate(ports, 1):
        print(f"  {i}. {p.device} - {p.description}")


def find_valid_port(slave_id: int = 1, baudrate: int = 115200, timeout: float = 1.0):
    """返回第一个能响应 Modbus 只读请求的串口。"""
    for port_info in serial.tools.list_ports.comports():
        sdk = None
        try:
            sdk = Changingtek_rtu_psdk(port_info.device, slave_id, baudrate, timeout)
            if sdk.connect():
                sdk.ready()  # 读取只读寄存器 REG_READY，确认设备可通信
                return port_info.device
        except (minimalmodbus.ModbusException, serial.SerialException, OSError, ValueError):
            pass
        finally:
            if sdk:
                sdk.disconnect()
    return None


if __name__ == "__main__":
    """
    Changingtek_rtu_psdk.py
    
    知行手闭环步进电机 Python SDK 使用示例
    
    本示例演示了如何使用 Changingtek_rtu_psdk 类控制执行器：
    1. 连接串口
    2. 使能执行器
    3. 执行临时位置控制 (移动到指定位置)
    4. 读取状态反馈 (位置、报警等)
    5. 多点运动模式示例 (注释状态)
    """

    # =========================================================================
    # 用户配置区域 (请根据实际情况修改以下参数)
    # =========================================================================
    
    # 1. 串口配置
    list_ports()
    PORT = find_valid_port()
    if not PORT:
        print("未发现可通信的夹爪串口，退出")
        exit(1)
    print(f"自动检测到有效串口: {PORT}")

    SLAVE_ID = 1
    
    # 2. 运动参数配置
    # 位置单位说明: 1 unit = 0.01 mm
    # 例如: 2000 units = 20.00 mm; 12000 units = 120.00 mm
    POS_OPEN = 0           # 打开位置 (0.00mm)
    POS_CLOSE = 12000      # 闭合位置 (120.00mm)。以CTAG2F120s为例，行程120mm，范围0-12000，以此类推。
    
    SPEED = 100            # 速度 % (0-100)
    FORCE = 60             # 力矩/电流 % (0-100)
    
    # 3. 循环测试配置
    LOOP_COUNT = 5         # 自动循环次数

    # =========================================================================
    # 配置结束
    # =========================================================================

    sdk = Changingtek_rtu_psdk(PORT, SLAVE_ID, baudrate=115200, timeout=1.0)

    try:
        print(f"Connecting to {PORT} ...")
        if sdk.connect():
            print("Connected.")
        else:
            print("Connection failed.")
            exit(1)
        # 1. 使能执行器
        print("Enabling actuator...")
        sdk.enable(True)
        
        print(f"Starting loop control ({LOOP_COUNT} times)...")
        print("Press Ctrl+C to stop")
        
        for i in range(LOOP_COUNT):
            print(f"\n=== Loop {i + 1} / {LOOP_COUNT} ===")
            
            # --- 动作 1: 打开 (移动到 POS_OPEN) ---
            print(f"[Open] Moving to {POS_OPEN} ...")
            sdk.temp_move(POS_OPEN, SPEED, FORCE, 100, 100, True)
            
            # 等待到位
            res_open = sdk.wait_until_pos_or_torque(10.0)
            fb_open = sdk.feedback_position()
            print(f"  -> Result: {res_open} [Pos Feedback: {fb_open}]")
            
            # --- 动作 2: 闭合 (移动到 POS_CLOSE) ---
            print(f"[Close] Moving to {POS_CLOSE} ...")
            sdk.temp_move(POS_CLOSE, SPEED, FORCE, 100, 100, True)
            
            # 等待到位 (如果是夹取物体，可能会触发 torque 力控到位)
            res_close = sdk.wait_until_pos_or_torque(10.0)
            fb_close = sdk.feedback_position()
            is_torque_reached = sdk.torque_reached()
            print(f"  -> Result: {res_close} [Pos Feedback: {fb_close}] [Torque Reached: {'Yes' if is_torque_reached else 'No'}]")

        print("\nLoop finished. Executing final open...")
        
        # --- 最终动作: 打开 (移动到 POS_OPEN) ---
        print(f"[Final Open] Moving to {POS_OPEN} ...")
        sdk.temp_move(POS_OPEN, SPEED, FORCE, 100, 100, True)
        
        # 等待到位
        res_final = sdk.wait_until_pos_or_torque(10.0)
        fb_final = sdk.feedback_position()
        print(f"  -> Result: {res_final} [Pos Feedback: {fb_final}]")
        
        # 读取最终状态
        print(f"Final Position: {sdk.feedback_position()}")
        print(f"Alarm bits    : {hex(sdk.read_alarm())}")

    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print("Error:", e)
