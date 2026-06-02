#!/usr/bin/env python3
"""
MLP推理速度测试脚本
测试MLPReasoner的推理性能，使用12维随机输入数据
"""

import os
import sys
import time
import numpy as np
import statistics
from typing import List

# 添加模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from reason import load_mlp_reasoner
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"无法导入推理器: {e}")
    TORCH_AVAILABLE = False

def generate_random_12d_input():
    """生成12维随机输入数据"""
    # 生成符合触觉传感器范围的随机数据
    # 力的范围: -50N 到 50N
    # 力矩的范围: -5Nm 到 5Nm
    
    force_l = np.random.uniform(-50, 50, 3).astype(np.float32)  # 左手力 (3,)
    force_r = np.random.uniform(-50, 50, 3).astype(np.float32)  # 右手力 (3,)
    moment_l = np.random.uniform(-5, 5, 3).astype(np.float32)   # 左手力矩 (3,)
    moment_r = np.random.uniform(-5, 5, 3).astype(np.float32)   # 右手力矩 (3,)
    
    return force_l, force_r, moment_l, moment_r

def test_single_inference(reasoner, verbose=False):
    """测试单次推理"""
    force_l, force_r, moment_l, moment_r = generate_random_12d_input()
    
    if verbose:
        print(f"输入数据:")
        print(f"  左手力: {force_l}")
        print(f"  右手力: {force_r}")
        print(f"  左手力矩: {moment_l}")
        print(f"  右手力矩: {moment_r}")
    
    start_time = time.perf_counter()
    result = reasoner.predict(force_l, force_r, moment_l, moment_r)
    end_time = time.perf_counter()
    
    inference_time = (end_time - start_time) * 1000  # 转换为毫秒
    
    if verbose:
        print(f"  预测结果: {result}")
        print(f"  推理时间: {inference_time:.3f} ms")
    
    return inference_time, result

def test_batch_inference(reasoner, batch_size=100):
    """测试批量推理"""
    print(f"\n正在测试批量推理 (批次大小: {batch_size})...")
    
    # 生成批量数据
    batch_data = {
        'resultant_force_l': np.random.uniform(-50, 50, (batch_size, 3)).astype(np.float32),
        'resultant_force_r': np.random.uniform(-50, 50, (batch_size, 3)).astype(np.float32),
        'resultant_moment_l': np.random.uniform(-5, 5, (batch_size, 3)).astype(np.float32),
        'resultant_moment_r': np.random.uniform(-5, 5, (batch_size, 3)).astype(np.float32)
    }
    
    start_time = time.perf_counter()
    results = reasoner.predict_batch(batch_data)
    end_time = time.perf_counter()
    
    total_time = (end_time - start_time) * 1000  # 毫秒
    avg_time_per_sample = total_time / batch_size
    
    print(f"  批量推理总时间: {total_time:.3f} ms")
    print(f"  平均每样本时间: {avg_time_per_sample:.3f} ms")
    print(f"  吞吐量: {1000/avg_time_per_sample:.1f} samples/second")
    print(f"  输出形状: {results.shape}")
    
    return total_time, avg_time_per_sample

def benchmark_inference_speed(model_root_dir: str, num_trials: int = 1000):
    """基准测试推理速度"""
    if not TORCH_AVAILABLE:
        print("PyTorch不可用，无法进行推理测试")
        return
    
    print("=" * 60)
    print("MLP推理速度基准测试")
    print("=" * 60)
    
    try:
        # 加载推理器
        print(f"正在加载模型: {model_root_dir}")
        reasoner = load_mlp_reasoner(model_root_dir, device='cpu')
        reasoner.print_info()
        
        # 预热推理器
        print(f"\n正在预热推理器...")
        for _ in range(10):
            test_single_inference(reasoner, verbose=False)
        
        # 单次推理基准测试
        print(f"\n正在进行单次推理基准测试 ({num_trials} 次)...")
        inference_times = []
        
        for i in range(num_trials):
            if i % 100 == 0:
                print(f"  进度: {i}/{num_trials}")
            
            inference_time, _ = test_single_inference(reasoner, verbose=False)
            inference_times.append(inference_time)
        
        # 统计结果
        print(f"\n单次推理性能统计:")
        print(f"  总测试次数: {num_trials}")
        print(f"  平均推理时间: {statistics.mean(inference_times):.3f} ms")
        print(f"  中位数推理时间: {statistics.median(inference_times):.3f} ms")
        print(f"  最快推理时间: {min(inference_times):.3f} ms")
        print(f"  最慢推理时间: {max(inference_times):.3f} ms")
        print(f"  标准差: {statistics.stdev(inference_times):.3f} ms")
        print(f"  吞吐量: {1000/statistics.mean(inference_times):.1f} samples/second")
        
        # 批量推理测试
        for batch_size in [10, 50, 100, 500]:
            test_batch_inference(reasoner, batch_size)
        
        # 显示几个详细的推理示例
        print(f"\n详细推理示例:")
        for i in range(3):
            print(f"\n示例 {i+1}:")
            test_single_inference(reasoner, verbose=True)
        
        print("\n" + "=" * 60)
        print("基准测试完成!")
        print("=" * 60)
        
    except Exception as e:
        print(f"测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

def test_with_specific_model():
    """使用指定模型进行测试"""
    # 使用现有的模型目录
    test_model_dirs = [
        "/home/lyj/robot_space_2/ros2_driver_layer/src/haptic/haptic/models",
        "./models",
        "./models/mlp_policy",
        "./models/mlp_policy_genrecttri"
    ]
    
    for model_dir in test_model_dirs:
        if os.path.exists(model_dir):
            print(f"找到模型目录: {model_dir}")
            benchmark_inference_speed(model_dir, num_trials=500)
            return
    
    print("未找到可用的模型目录，请提供正确的模型路径")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MLP推理速度测试")
    parser.add_argument("--model_dir", type=str, help="模型根目录路径")
    parser.add_argument("--trials", type=int, default=1000, help="测试次数")
    parser.add_argument("--quick", action="store_true", help="快速测试模式")
    
    args = parser.parse_args()
    
    if args.quick:
        args.trials = 100
    
    if args.model_dir:
        benchmark_inference_speed(args.model_dir, args.trials)
    else:
        test_with_specific_model()

if __name__ == "__main__":
    main()
