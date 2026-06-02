"""
时序策略学习数据集 - 基于PointPairDataset扩展
支持LSTM训练所需的时序数据加载
"""
import os
import torch
import numpy as np
import random
import hashlib
import json
from torch.utils.data import Dataset
from DatasetPointPair import PointPairDataset

class SequenceDataset(PointPairDataset):
    """
    时序策略学习数据集，扩展自PointPairDataset
    支持加载连续时序数据用于LSTM训练
    """
    
    def __init__(self, data_root, categories=None, is_train=True, use_resultant=True, use_forces=False,
                 normalization_config=None, sequence_length=10, prediction_step=1, overlap_stride=5):
        """
        Args:
            data_root: 数据根目录路径
            categories: 要包含的类别列表
            is_train: 是否加载训练集数据
            use_resultant: 是否使用resultant数据
            use_forces: 是否使用forces数据
            normalization_config: 归一化配置字典
            sequence_length: 时序序列长度 (LSTM输入的时间步数)
            prediction_step: 预测步长，预测序列后第prediction_step个时刻的动作
            overlap_stride: 序列重叠步长，控制滑动窗口的步长
        """
        self.sequence_length = sequence_length
        self.overlap_stride = overlap_stride
        
        # 调用父类初始化，但使用prediction_step=1（因为我们会重新构建索引）
        super().__init__(
            data_root=data_root, 
            categories=categories, 
            is_train=is_train,
            use_resultant=use_resultant,
            use_forces=use_forces,
            normalization_config=normalization_config,
            prediction_step=prediction_step
        )
        
        # 重新构建时序索引
        self._build_sequence_indices()
        
    def _build_sequence_indices(self):
        """构建时序序列索引 - 使用prediction_step作为全局采样间隔"""
        self.sequence_indices = []
        
        for traj_idx, traj_info in enumerate(self.trajectories):
            # 读取末端位置数据确定长度
            position_data = np.load(os.path.join(traj_info['path'], "_end_position.npy"))
            trajectory_length = len(position_data)
            
            # 新采样策略：使用prediction_step作为全局采样间隔T
            # 序列采样位置: [start, start+T, start+2T, ..., start+(sequence_length-1)*T]
            # 预测目标位置: start + sequence_length * T
            # 所以最小轨迹长度需要: start + sequence_length * prediction_step
            min_required_length = self.sequence_length * self.prediction_step + 1
            
            if trajectory_length < min_required_length:
                continue  # 跳过太短的轨迹
            
            # 滑动窗口生成序列
            # 序列起始位置的范围: [0, trajectory_length - min_required_length]
            max_start_idx = trajectory_length - min_required_length
            
            for start_idx in range(0, max_start_idx + 1, self.overlap_stride):
                # 计算序列中每个时间步的实际索引
                seq_indices = [start_idx + i * self.prediction_step for i in range(self.sequence_length)]
                target_idx = start_idx + self.sequence_length * self.prediction_step  # 预测目标位置
                
                self.sequence_indices.append({
                    'traj_idx': traj_idx,
                    'seq_indices': seq_indices,      # 序列中每个时间步的实际索引
                    'target_idx': target_idx,        # 目标动作索引
                    'seq_length': self.sequence_length,  # 序列长度
                    'sampling_interval': self.prediction_step  # 采样间隔
                })
        
        # 打乱序列索引
        random.seed(self.random_seed)
        random.shuffle(self.sequence_indices)
        
        print(f"时序模式 (序列长度={self.sequence_length}, 采样间隔={self.prediction_step}, 重叠步长={self.overlap_stride}): {len(self.sequence_indices)} 序列已加载。")
        
    def _print_dataset_info(self):
        """打印数据集信息"""
        total_trajectories = len(self.trajectories)
        total_sequences = len(getattr(self, 'sequence_indices', []))
        
        print(f"[SequenceDataset] 数据统计:")
        print(f"  - 轨迹数量: {total_trajectories}")
        print(f"  - 序列数量: {total_sequences}")
        print(f"  - 序列长度: {self.sequence_length}")
        print(f"  - 采样间隔: {self.prediction_step}")
        print(f"  - 重叠步长: {self.overlap_stride}")
        print(f"  - 当前集合: {'训练集' if self.is_train else '测试集'}")

    def __len__(self):
        return len(self.sequence_indices)

    def __getitem__(self, idx):
        """获取时序序列样本"""
        sequence_info = self.sequence_indices[idx]
        traj_info = self.trajectories[sequence_info['traj_idx']]
        traj_path = traj_info['path']
        
        seq_indices = sequence_info['seq_indices']  # 序列中每个时间步的实际索引
        target_idx = sequence_info['target_idx']
        
        return self._load_sequence_data(traj_path, traj_info, seq_indices, target_idx)

    def _load_sequence_data(self, traj_path, traj_info, seq_indices, target_idx):
        """
        加载时序序列数据 - 使用间隔采样，支持密集监督
        Args:
            seq_indices: 序列中每个时间步的实际索引列表 [start, start+T, start+2T, ...]
            target_idx: 目标动作索引
        Returns:
            result: 包含时序输入数据和目标动作的字典
        """
        # 加载末端位置数据
        position_data = np.load(os.path.join(traj_path, "_end_position.npy"))
        
        # 序列动作数据 (历史动作序列) - 按间隔采样
        action_seq = []
        for idx in seq_indices:
            action = position_data[idx, 1:4]  # (3,) XYZ坐标
            action_seq.append(self._normalize_data(action, 'actions'))
        action_seq = np.array(action_seq)  # (seq_len, 3)
        
        # 目标动作 (预测目标) - 单步预测
        target_action = position_data[target_idx, 1:4]  # (3,) XYZ坐标
        target_action = self._normalize_data(target_action, 'actions')
        
        # 密集监督的目标动作序列 - 所有时间步的目标动作
        # 构建目标序列：从seq_indices[0]+prediction_step开始的连续动作
        target_action_seq = []
        start_target_idx = seq_indices[0] + self.prediction_step
        for i in range(len(seq_indices)):
            target_seq_idx = start_target_idx + i * self.prediction_step
            if target_seq_idx < len(position_data):
                target_action_seq.append(self._normalize_data(position_data[target_seq_idx, 1:4], 'actions'))
            else:
                # 如果超出范围，使用最后一个有效动作
                target_action_seq.append(self._normalize_data(position_data[-1, 1:4], 'actions'))
        target_action_seq = np.array(target_action_seq)  # (seq_len, 3)
        
        result = {
            'action_seq': torch.FloatTensor(action_seq),          # (seq_len, 3) 历史动作序列
            'target_next_action': torch.FloatTensor(target_action),  # (3,) 目标动作（单步）
            'target_action_seq': torch.FloatTensor(target_action_seq),  # (seq_len, 3) 目标动作序列（密集监督）
            'category': traj_info['category'],
            'trajectory_id': traj_info['dir_name'],
            'seq_indices': seq_indices,  # 序列实际索引
            'target_idx': target_idx,
            'seq_length': len(seq_indices),
            'sampling_interval': self.prediction_step
        }
        
        # 加载时序触觉数据
        if self.use_resultant:
            # 加载resultants时序数据
            resultant_force_l_data = np.load(os.path.join(traj_path, "_resultant_force_l.npy"))
            resultant_force_r_data = np.load(os.path.join(traj_path, "_resultant_force_r.npy"))
            resultant_moment_l_data = np.load(os.path.join(traj_path, "_resultant_moment_l.npy"))
            resultant_moment_r_data = np.load(os.path.join(traj_path, "_resultant_moment_r.npy"))
            
            # 提取序列 - 按间隔采样
            resultant_force_l_seq = []
            resultant_force_r_seq = []
            resultant_moment_l_seq = []
            resultant_moment_r_seq = []
            
            for idx in seq_indices:
                resultant_force_l_seq.append(self._normalize_data(resultant_force_l_data[idx], 'resultants'))
                resultant_force_r_seq.append(self._normalize_data(resultant_force_r_data[idx], 'resultants'))
                resultant_moment_l_seq.append(self._normalize_data(resultant_moment_l_data[idx], 'resultants'))
                resultant_moment_r_seq.append(self._normalize_data(resultant_moment_r_data[idx], 'resultants'))
            
            # 转换为numpy数组后再创建tensor (性能优化)
            result['resultant_force_l_seq'] = torch.FloatTensor(np.array(resultant_force_l_seq))    # (seq_len, 3)
            result['resultant_force_r_seq'] = torch.FloatTensor(np.array(resultant_force_r_seq))    # (seq_len, 3)
            result['resultant_moment_l_seq'] = torch.FloatTensor(np.array(resultant_moment_l_seq))  # (seq_len, 3)
            result['resultant_moment_r_seq'] = torch.FloatTensor(np.array(resultant_moment_r_seq))  # (seq_len, 3)
        
        if self.use_forces:
            # 加载forces时序数据
            forces_l_data = np.load(os.path.join(traj_path, "_forces_l.npy"))
            forces_r_data = np.load(os.path.join(traj_path, "_forces_r.npy"))
            
            # 提取序列并归一化 - 按间隔采样
            forces_l_seq = []
            forces_r_seq = []
            
            for idx in seq_indices:
                forces_l_seq.append(self._normalize_data(forces_l_data[idx], 'forces'))
                forces_r_seq.append(self._normalize_data(forces_r_data[idx], 'forces'))
            
            # 转换为numpy数组后再创建tensor (性能优化)
            result['forces_l_seq'] = torch.FloatTensor(np.array(forces_l_seq))  # (seq_len, 3, 20, 20)
            result['forces_r_seq'] = torch.FloatTensor(np.array(forces_r_seq))  # (seq_len, 3, 20, 20)
        
        return result


def create_sequence_datasets(data_root, categories=None, normalization_config=None, 
                           sequence_length=10, prediction_step=1, overlap_stride=5,
                           use_forces=True):
    """
    创建时序训练集和测试集
    Args:
        sequence_length: 时序序列长度
        prediction_step: 预测步长
        overlap_stride: 序列重叠步长
        use_forces: 是否使用forces数据（LSTM通常需要完整的触觉数据）
    """
    # 1. 先创建训练集
    train_dataset = SequenceDataset(
        data_root=data_root,
        categories=categories,
        is_train=True,
        use_forces=use_forces,  # LSTM通常使用完整的触觉数据
        use_resultant=False,     # 也可以使用resultant数据
        normalization_config=normalization_config,
        sequence_length=sequence_length,
        prediction_step=prediction_step,
        overlap_stride=overlap_stride
    )
    
    # 2. 创建测试集，使用训练集的归一化参数
    test_dataset = SequenceDataset(
        data_root=data_root,
        categories=categories,
        is_train=False,
        use_forces=use_forces,
        use_resultant=False,
        normalization_config=train_dataset.normalization_config,
        sequence_length=sequence_length,
        prediction_step=prediction_step,
        overlap_stride=overlap_stride
    )
    
    print(f"📊 时序数据集信息:")
    action_params = train_dataset.normalization_config.get('actions', {})
    forces_params = train_dataset.normalization_config.get('forces', {})
    print(f"   Actions归一化: {action_params.get('method', 'None')}")
    print(f"   Forces归一化: {forces_params.get('method', 'None')}")
    print(f"   测试集使用预计算参数: {test_dataset.use_precomputed_normalization}")
    
    return train_dataset, test_dataset, train_dataset.get_normalization_params()


# 自定义collate函数处理变长序列（可选）
def sequence_collate_fn(batch):
    """
    自定义collate函数，支持变长序列（如果需要的话）
    目前假设所有序列长度相同，可以扩展支持变长
    """
    # 当前实现：假设所有序列长度相同
    return torch.utils.data.dataloader.default_collate(batch)


if __name__ == '__main__':
    # 测试时序数据集
    print("🧪 测试时序数据集...")
    
    train_dataset, test_dataset, norm_config = create_sequence_datasets(
        data_root='data25.7_aligned',
        categories=['cir_lar'],  # 测试用小数据集
        sequence_length=8,
        prediction_step=1,
        overlap_stride=4,
        use_forces=True
    )
    
    print(f"\n📊 数据集大小:")
    print(f"  训练集: {len(train_dataset)} 序列")
    print(f"  测试集: {len(test_dataset)} 序列")
    
    # 测试数据加载
    sample = train_dataset[0]
    print(f"\n📦 样本数据结构:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {value}")
    
    # 测试DataLoader
    from torch.utils.data import DataLoader
    loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=sequence_collate_fn)
    
    batch = next(iter(loader))
    print(f"\n📦 批次数据结构:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)} (长度: {len(value)})")
    
    print("✅ 时序数据集测试完成！")
