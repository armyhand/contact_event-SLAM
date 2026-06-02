"""
LSTM策略测试脚本
模拟触觉数据和位姿数据输入，实时可视化网络输出的增量
直接用Python运行，不依赖ROS环境
"""

import os
import sys
import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import time

# 添加模型路径
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'models', 'feature_lstm')
sys.path.insert(0, model_path)

# 导入LSTM模型
from feature_lstm import TactilePolicyFeatureLSTM


class LSTMTester:
    """LSTM策略测试器"""
    
    def __init__(self):
        # 模型路径配置
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = os.path.join(current_dir, 'models', 'feature_lstm')
        self.model_path = os.path.join(self.model_dir, 'best_model.pt')
        self.config_path = os.path.join(self.model_dir, 'config.yaml')
        
        # 加载配置
        self._load_config()
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载模型
        self._load_model()
        
        # 序列长度
        self.sequence_length = self.config['data']['sequence_length']
        
        # 数据队列
        self.force_l_queue = deque(maxlen=self.sequence_length)
        self.force_r_queue = deque(maxlen=self.sequence_length)
        self.pose_xyz_queue = deque(maxlen=self.sequence_length)
        
        # 当前位姿 (初始化为工作空间中心)
        self.current_pose_xyz = np.array([0.6, 0.15, 0.3], dtype=np.float32)
        
        # 增量放大因子
        self.delta_scale_x = 1.0
        self.delta_scale_y = 2.0
        self.delta_scale_z = 1.0
        
        # 用于可视化的历史数据
        self.history_length = 100
        self.time_history = deque(maxlen=self.history_length)
        self.delta_x_history = deque(maxlen=self.history_length)
        self.delta_y_history = deque(maxlen=self.history_length)
        self.delta_z_history = deque(maxlen=self.history_length)
        self.delta_scaled_x_history = deque(maxlen=self.history_length)
        self.delta_scaled_y_history = deque(maxlen=self.history_length)
        self.delta_scaled_z_history = deque(maxlen=self.history_length)
        
        self.start_time = time.time()
        self.inference_count = 0
        
        print(f"LSTM测试器初始化完成")
        print(f"序列长度: {self.sequence_length}")
        print(f"增量放大因子: DX={self.delta_scale_x}, DY={self.delta_scale_y}, DZ={self.delta_scale_z}")
    
    def _load_config(self):
        """加载配置文件"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.config = {
            'model': config['model']['value'],
            'data': config['data']['value'],
        }
        
        # 归一化参数
        computed_norm = config.get('computed_normalization_params', {}).get('value', {})
        
        if not computed_norm:
            computed_norm = self.config['data'].get('normalization_config', {})
        
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
        
        print("配置文件加载成功")
    
    def _load_model(self):
        """加载训练好的LSTM模型"""
        model_config = self.config['model']
        
        # CNN编码器路径 (使用绝对路径)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cnn_encoder_path = os.path.join(current_dir, 'models', 'cnn_ae', 'best_model.pt')
        
        self.model = TactilePolicyFeatureLSTM(
            feature_dim=model_config['feature_dim'],
            action_dim=model_config['action_dim'],
            lstm_hidden_dim=model_config['lstm_hidden_dim'],
            lstm_num_layers=model_config['lstm_num_layers'],
            dropout_rate=model_config['dropout_rate'],
            pretrained_encoder_path=cnn_encoder_path,
            action_embed_dim=model_config['action_embed_dim'],
            fc_hidden_dims=model_config['fc_hidden_dims'],
        )
        
        # 加载权重
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"加载模型checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
            else:
                self.model.load_state_dict(checkpoint)
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"模型加载成功: {self.model_path}")
    
    def _normalize_forces(self, forces):
        """归一化触觉力数据"""
        mean = self.norm_params['forces']['params']['mean']
        std = self.norm_params['forces']['params']['std']
        return (forces - mean) / std
    
    def _normalize_pose(self, pose_xyz):
        """归一化位姿XYZ数据"""
        mean = self.norm_params['action']['params']['mean']
        std = self.norm_params['action']['params']['std']
        return (pose_xyz - mean) / std
    
    def _denormalize_pose(self, normalized_pose):
        """反归一化位姿XYZ数据"""
        mean = self.norm_params['action']['params']['mean']
        std = self.norm_params['action']['params']['std']
        return normalized_pose * std + mean
    
    def generate_random_force_data(self):
        """生成随机触觉力数据 (3, 20, 20)"""
        # 模拟真实触觉数据分布: 小幅度随机噪声
        force_data = np.random.randn(3, 20, 20).astype(np.float32) * 0.02
        return force_data
    
    def update_pose(self):
        """更新位姿 (模拟小幅度移动)"""
        # 添加小的随机扰动
        noise = np.random.randn(3).astype(np.float32) * 0.001
        self.current_pose_xyz += noise
        
        # 限制在工作空间内
        self.current_pose_xyz[0] = np.clip(self.current_pose_xyz[0], 0.4, 0.8)
        self.current_pose_xyz[1] = np.clip(self.current_pose_xyz[1], -0.3, 0.3)
        self.current_pose_xyz[2] = np.clip(self.current_pose_xyz[2], 0.2, 0.5)
    
    def inference_step(self):
        """执行一次推理"""
        # 生成随机数据
        force_l = self.generate_random_force_data()
        force_r = self.generate_random_force_data()
        
        # 更新位姿
        self.update_pose()
        
        # 添加到队列
        self.force_l_queue.append(force_l)
        self.force_r_queue.append(force_r)
        self.pose_xyz_queue.append(self.current_pose_xyz.copy())
        
        # 检查队列是否已满
        if len(self.force_l_queue) < self.sequence_length:
            return None
        
        # 准备输入数据
        forces_l_seq = np.array(list(self.force_l_queue), dtype=np.float32)
        forces_r_seq = np.array(list(self.force_r_queue), dtype=np.float32)
        pose_xyz_seq = np.array(list(self.pose_xyz_queue), dtype=np.float32)
        
        # 归一化
        forces_l_seq = self._normalize_forces(forces_l_seq)
        forces_r_seq = self._normalize_forces(forces_r_seq)
        pose_xyz_seq_norm = self._normalize_pose(pose_xyz_seq)
        
        # 转换为张量
        forces_l_tensor = torch.from_numpy(forces_l_seq).unsqueeze(0).to(self.device)
        forces_r_tensor = torch.from_numpy(forces_r_seq).unsqueeze(0).to(self.device)
        pose_xyz_tensor = torch.from_numpy(pose_xyz_seq_norm).unsqueeze(0).to(self.device)
        
        # 模型推理
        with torch.no_grad():
            predicted_pose_norm = self.model(forces_l_tensor, forces_r_tensor, pose_xyz_tensor)
            predicted_pose_norm = predicted_pose_norm.squeeze(0).cpu().numpy()
            
            # 反归一化得到预测位姿
            predicted_pose_xyz = self._denormalize_pose(predicted_pose_norm)
            
            # 计算增量
            delta_xyz = predicted_pose_xyz - self.current_pose_xyz
            
            # 应用放大因子
            delta_xyz_scaled = np.array([
                delta_xyz[0] * self.delta_scale_x,
                delta_xyz[1] * self.delta_scale_y,
                delta_xyz[2] * self.delta_scale_z
            ], dtype=np.float32)
        
        # 记录历史数据
        current_time = time.time() - self.start_time
        self.time_history.append(current_time)
        self.delta_x_history.append(delta_xyz[0])
        self.delta_y_history.append(delta_xyz[1])
        self.delta_z_history.append(delta_xyz[2])
        self.delta_scaled_x_history.append(delta_xyz_scaled[0])
        self.delta_scaled_y_history.append(delta_xyz_scaled[1])
        self.delta_scaled_z_history.append(delta_xyz_scaled[2])
        
        self.inference_count += 1
        
        return {
            'current_pose': self.current_pose_xyz.copy(),
            'predicted_pose': predicted_pose_xyz,
            'delta': delta_xyz,
            'delta_scaled': delta_xyz_scaled,
            'time': current_time
        }
    
    def run_visualization(self):
        """运行可视化"""
        # 创建图形
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle('LSTM Policy Delta Output Test (Real-time)', fontsize=16)
        
        # 初始化空线条
        line_x_raw, = axes[0].plot([], [], 'b-', label='Raw Delta', linewidth=2)
        line_x_scaled, = axes[0].plot([], [], 'r--', label='Scaled Delta', linewidth=2)
        axes[0].axhline(y=0, color='g', linestyle='-', linewidth=1.5, alpha=0.7)  # 零线
        axes[0].set_ylabel('ΔX (m)', fontsize=12)
        axes[0].set_title(f'X-axis Delta (Scale Factor: {self.delta_scale_x}x)', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        line_y_raw, = axes[1].plot([], [], 'b-', label='Raw Delta', linewidth=2)
        line_y_scaled, = axes[1].plot([], [], 'r--', label='Scaled Delta', linewidth=2)
        axes[1].axhline(y=0, color='g', linestyle='-', linewidth=1.5, alpha=0.7)  # 零线
        axes[1].set_ylabel('ΔY (m)', fontsize=12)
        axes[1].set_title(f'Y-axis Delta (Scale Factor: {self.delta_scale_y}x)', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        line_z_raw, = axes[2].plot([], [], 'b-', label='Raw Delta', linewidth=2)
        line_z_scaled, = axes[2].plot([], [], 'r--', label='Scaled Delta', linewidth=2)
        axes[2].axhline(y=0, color='g', linestyle='-', linewidth=1.5, alpha=0.7)  # 零线
        axes[2].set_xlabel('Time (s)', fontsize=12)
        axes[2].set_ylabel('ΔZ (m)', fontsize=12)
        axes[2].set_title(f'Z-axis Delta (Scale Factor: {self.delta_scale_z}x)', fontsize=12)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # 更新函数
        def update(frame):
            # 执行推理
            result = self.inference_step()
            
            if result is None:
                return line_x_raw, line_x_scaled, line_y_raw, line_y_scaled, line_z_raw, line_z_scaled
            
            # 更新数据
            times = list(self.time_history)
            
            # X轴
            line_x_raw.set_data(times, list(self.delta_x_history))
            line_x_scaled.set_data(times, list(self.delta_scaled_x_history))
            
            # Y轴
            line_y_raw.set_data(times, list(self.delta_y_history))
            line_y_scaled.set_data(times, list(self.delta_scaled_y_history))
            
            # Z轴
            line_z_raw.set_data(times, list(self.delta_z_history))
            line_z_scaled.set_data(times, list(self.delta_scaled_z_history))
            
            # 自动调整坐标轴范围
            if len(times) > 0:
                for ax in axes:
                    ax.set_xlim(max(0, times[-1] - 10), times[-1] + 1)
                
                # 动态调整Y轴范围
                all_deltas_x = list(self.delta_x_history) + list(self.delta_scaled_x_history)
                all_deltas_y = list(self.delta_y_history) + list(self.delta_scaled_y_history)
                all_deltas_z = list(self.delta_z_history) + list(self.delta_scaled_z_history)
                
                if all_deltas_x:
                    y_range_x = max(abs(min(all_deltas_x)), abs(max(all_deltas_x))) * 1.2
                    axes[0].set_ylim(-y_range_x, y_range_x)
                
                if all_deltas_y:
                    y_range_y = max(abs(min(all_deltas_y)), abs(max(all_deltas_y))) * 1.2
                    axes[1].set_ylim(-y_range_y, y_range_y)
                
                if all_deltas_z:
                    y_range_z = max(abs(min(all_deltas_z)), abs(max(all_deltas_z))) * 1.2
                    axes[2].set_ylim(-y_range_z, y_range_z)
            
            # 打印统计信息
            if self.inference_count % 10 == 0:
                print(f"\n推理次数: {self.inference_count}")
                print(f"当前位姿: [{result['current_pose'][0]:.6f}, {result['current_pose'][1]:.6f}, {result['current_pose'][2]:.6f}]")
                print(f"原始增量: [{result['delta'][0]:.6f}, {result['delta'][1]:.6f}, {result['delta'][2]:.6f}]")
                print(f"放大增量: [{result['delta_scaled'][0]:.6f}, {result['delta_scaled'][1]:.6f}, {result['delta_scaled'][2]:.6f}]")
            
            return line_x_raw, line_x_scaled, line_y_raw, line_y_scaled, line_z_raw, line_z_scaled
        
        # 创建动画 (10Hz更新频率)
        anim = FuncAnimation(fig, update, interval=100, blit=True, cache_frame_data=False)
        
        plt.tight_layout()
        plt.show()


def main():
    print("=== LSTM策略测试器 ===")
    print("生成随机触觉数据和位姿数据，实时可视化网络输出增量")
    print()
    
    try:
        tester = LSTMTester()
        print("\n开始测试，显示实时增量输出...")
        print("关闭窗口以停止测试")
        print()
        
        tester.run_visualization()
        
    except KeyboardInterrupt:
        print("\n测试已停止")
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
