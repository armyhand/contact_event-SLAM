"""
Feature-LSTM模型 - 时序版本（增量预测）
# 探索dui
输入: 时序触觉数据 forces_l[seq_len, 3, 20, 20] + forces_r[seq_len, 3, 20, 20] -> CNN特征提取 -> LSTM -> 预测动作增量
输出: action_delta[3] = 3维动作增量，最终输出 = current_action + action_delta
注意: 
1. 网络预测增量而非绝对动作，提高预测稳定性
2. 损失计算使用转换后的绝对预测，保持现有逻辑不变
3. 支持密集监督模式的累积增量预测
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 导入CNN自编码器
try:
    from haptic.models.cnn_ae.cnn_autoencoder import TactileCNNAutoencoder
except ImportError:
    # 开发环境下的导入
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, project_root)
    from cnn_ae.cnn_autoencoder import TactileCNNAutoencoder



class TactilePolicyFeatureLSTM(nn.Module):
    """
    触觉策略Feature-LSTM模型 - 时序预测版本（增量预测）
    基于预训练触觉特征的时序增量预测
    
    架构：
    1. 预训练CNN编码器提取左右手触觉特征 (128维 × 2 = 256维)
    2. 动作嵌入编码器 (3 → action_embed_dim)
    3. 拼接特征和动作 (256 + action_embed_dim)
    4. 通过LSTM处理时序信息
    5. 全连接层预测下一时刻动作增量
    6. 增量 + 当前动作 = 绝对预测（用于损失计算）
    
    增量预测优势：
    - 更稳定的预测，避免大幅度跳跃
    - 符合控制系统直觉（小幅调整）
    - 网络学习相对变化，而非绝对位置
    - 损失计算逻辑保持不变
    """
    
    def __init__(self, 
                 feature_dim=128,           # 单手特征维度
                 action_dim=3,              # 输出动作维度 (dx, dy, dz)
                 lstm_hidden_dim=256,       # LSTM隐藏维度
                 lstm_num_layers=2,         # LSTM层数
                 dropout_rate=0.25,         # Dropout比率
                 pretrained_encoder_path=None,
                 action_embed_dim=64,       # 动作嵌入维度
                 fc_hidden_dims=[128, 64],  # 最后全连接层维度
                 ):
        """
        Args:
            feature_dim: 单手触觉特征维度
            action_dim: 输出动作维度
            lstm_hidden_dim: LSTM隐藏状态维度
            lstm_num_layers: LSTM层数
            dropout_rate: Dropout比率
            pretrained_encoder_path: 预训练编码器权重路径
            action_embed_dim: 动作嵌入维度
            fc_hidden_dims: 最后全连接层的隐藏维度列表
        """
        super(TactilePolicyFeatureLSTM, self).__init__()
        
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.action_embed_dim = action_embed_dim
        
        # 加载预训练的触觉特征提取器
        self.tactile_encoder = TactileCNNAutoencoder(
            in_channels=3, 
            latent_dim=feature_dim
        )
        
        # 加载预训练权重
        if pretrained_encoder_path is not None and os.path.exists(pretrained_encoder_path):
            print(f"加载预训练触觉编码器: {pretrained_encoder_path}")
            checkpoint = torch.load(pretrained_encoder_path, map_location='cpu', weights_only=False)
            
            # 检查checkpoint格式，提取模型状态字典
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model_state = checkpoint['model_state_dict']
                    print("📦 检测到训练checkpoint格式，提取model_state_dict")
                elif 'state_dict' in checkpoint:
                    model_state = checkpoint['state_dict']
                    print("📦 检测到state_dict格式")
                else:
                    model_state = checkpoint
                    print("📦 检测到直接状态字典格式")
            else:
                model_state = checkpoint
            
            # 加载状态字典
            self.tactile_encoder.load_state_dict(model_state, strict=True)
            print("✅ 成功加载预训练权重")
            
            # 打印checkpoint信息
            if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
                print(f"📊 预训练模型信息: epoch {checkpoint['epoch']}")
                    
            
            # 冻结特征提取器参数
            for param in self.tactile_encoder.parameters():
                param.requires_grad = False
            print("🔒 特征提取器参数已冻结")
        else:
            print("❌ 无法导入CNN编码器")
            raise FileNotFoundError(f"预训练编码器路径无效: {pretrained_encoder_path}")
        
        # 动作嵌入编码器
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, self.action_embed_dim),
            nn.LayerNorm(self.action_embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # LSTM输入维度: 左右手特征 + 动作嵌入
        lstm_input_dim = feature_dim * 2 + self.action_embed_dim  # 256 + 64 = 320
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            dropout=dropout_rate if lstm_num_layers > 1 else 0,
            batch_first=True  # 输入格式: (batch, seq, feature)
        )
        
        # 全连接层
        fc_layers = []
        prev_dim = lstm_hidden_dim
        
        for hidden_dim in fc_hidden_dims:
            fc_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        fc_layers.append(nn.Linear(prev_dim, action_dim))
        
        self.fc = nn.Sequential(*fc_layers)
        
        # 初始化权重
        self._initialize_weights()
        
        # 统计参数
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")
        
    def _initialize_weights(self):
        """初始化LSTM权重"""
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                if param.dim() >= 2:  # 只对2维以上的权重应用xavier初始化
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.normal_(param, 0, 0.01)  # 对1维权重使用正态分布初始化
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
        # 初始化输出层权重
        for module in self.fc.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, forces_l_seq, forces_r_seq, action_seq, seq_lengths=None, return_all_steps=False):
        """
        前向传播
        
        Args:
            forces_l_seq: 左手触觉力时序数据 (B, T, 3, 20, 20)
            forces_r_seq: 右手触觉力时序数据 (B, T, 3, 20, 20)
            action_seq: 动作时序数据 (B, T, 3)
            seq_lengths: 每个序列的实际长度 (B,) 可选，用于变长序列
            return_all_steps: 是否返回所有时间步的预测 (用于密集监督)
            
        Returns:
            如果 return_all_steps=False: next_action (B, 3) - 仅最后时间步预测
            如果 return_all_steps=True: all_predictions (B, T, 3) - 所有时间步预测
        """
        batch_size, seq_len = forces_l_seq.size(0), forces_l_seq.size(1)
        
        # 重塑为 (B*T, 3, 20, 20) 以便CNN处理
        forces_l_flat = forces_l_seq.view(-1, 3, 20, 20)  # (B*T, 3, 20, 20)
        forces_r_flat = forces_r_seq.view(-1, 3, 20, 20)  # (B*T, 3, 20, 20)
        
        if self.tactile_encoder is not None:
            # 使用预训练编码器提取特征
            features_l_flat = self.tactile_encoder.encoder(forces_l_flat)  # (B*T, feature_dim)
            features_r_flat = self.tactile_encoder.encoder(forces_r_flat)  # (B*T, feature_dim)
        else:
            return None
        
        # 重塑回时序格式
        features_l_seq = features_l_flat.view(batch_size, seq_len, self.feature_dim)  # (B, T, feature_dim)
        features_r_seq = features_r_flat.view(batch_size, seq_len, self.feature_dim)  # (B, T, feature_dim)
        
        # 动作嵌入
        action_embed_seq = self.action_encoder(action_seq)  # (B, T, action_embed_dim)
        
        # 拼接特征: [features_l, features_r, action_embed]
        combined_features = torch.cat([
            features_l_seq, 
            features_r_seq, 
            action_embed_seq
        ], dim=-1)  # (B, T, 256 + action_embed_dim)
        
        # LSTM处理时序信息
        if seq_lengths is not None:
            # 处理变长序列
            packed_input = nn.utils.rnn.pack_padded_sequence(
                combined_features, seq_lengths, batch_first=True, enforce_sorted=False
            )
            packed_output, (hidden, cell) = self.lstm(packed_input)
            lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        else:
            # 固定长度序列
            lstm_output, (hidden, cell) = self.lstm(combined_features)  # (B, T, lstm_hidden_dim)
        
        if return_all_steps:
            # 返回所有时间步的预测 (用于密集监督)
            # 重塑LSTM输出以便批量处理 - 使用reshape而不是view避免内存布局问题
            lstm_flat = lstm_output.reshape(-1, self.lstm_hidden_dim)  # (B*T, lstm_hidden_dim)
            
            # 通过全连接层得到所有时间步的增量预测
            delta_predictions_flat = self.fc(lstm_flat)  # (B*T, action_dim) - 增量预测
            
            # 重塑回时序格式
            delta_predictions = delta_predictions_flat.reshape(batch_size, seq_len, self.action_dim)  # (B, T, action_dim)
            
            # 将增量预测转换为绝对预测
            # 对于第一个时间步，使用输入动作的第一个时间步作为基准
            # 对于后续时间步，使用前一个时间步的预测作为基准
            all_predictions = torch.zeros_like(delta_predictions)
            
            for t in range(seq_len):
                if t == 0:
                    # 第一个时间步：预测 = 当前动作 + 增量
                    all_predictions[:, t, :] = action_seq[:, t, :] + delta_predictions[:, t, :]
                else:
                    # 后续时间步：预测 = 前一步预测 + 增量
                    all_predictions[:, t, :] = all_predictions[:, t-1, :] + delta_predictions[:, t, :]
            
            return all_predictions
        else:
            # 仅返回最后时间步的预测 (原始行为)
            if seq_lengths is not None:
                # 取每个序列的最后一个有效输出
                batch_indices = torch.arange(batch_size, device=lstm_output.device)
                last_outputs = lstm_output[batch_indices, seq_lengths - 1]  # (B, lstm_hidden_dim)
            else:
                last_outputs = lstm_output[:, -1, :]  # 取最后一个时间步 (B, lstm_hidden_dim)
            
            # 全连接层预测下一时刻动作增量
            delta_action = self.fc(last_outputs)  # (B, action_dim) - 预测增量
            
            # 将增量预测转换为绝对预测（加上当前动作）
            # 取最后一个时间步的动作作为基准
            if seq_lengths is not None:
                batch_indices = torch.arange(batch_size, device=action_seq.device)
                current_action = action_seq[batch_indices, seq_lengths - 1]  # (B, 3)
            else:
                current_action = action_seq[:, -1, :]  # (B, 3)
            
            next_action = current_action + delta_action  # 绝对预测 = 当前动作 + 增量
            
            return next_action
    
    def init_hidden(self, batch_size, device):
        """初始化LSTM隐藏状态"""
        hidden = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_dim).to(device)
        cell = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_dim).to(device)
        return (hidden, cell)


def compute_feature_lstm_losses(inputs, outputs, dataset=None, dense_supervision=False, current_epoch=1, total_epochs=100):
    """
    计算Feature-LSTM损失 - 支持密集监督和时序预测，包含步长范数权重和方向一致性损失
    
    Args:
        inputs: 输入数据字典，包含 'target_next_action' 或 'target_action_seq'
        outputs: 模型输出张量 
                 如果dense_supervision=False: (B, 3) - 下一时刻预测动作
                 如果dense_supervision=True: (B, T, 3) - 所有时间步预测动作
        dataset: 数据集对象，用于反归一化计算真实损失
        dense_supervision: 是否使用密集监督 (每个时间步都计算损失)
        current_epoch: 当前训练轮数 (从1开始)
        total_epochs: 总训练轮数
        
    Returns:
        loss: 总损失
        metrics: 损失分解字典
    """
    if dense_supervision:
        # 密集监督模式：对所有时间步计算损失
        predicted_action_seq = outputs  # (B, T, 3)
        target_action_seq = inputs['target_action_seq']  # (B, T, 3)
        
        # 计算步长范数 (B, T)
        predicted_step_norms = torch.norm(predicted_action_seq, dim=-1)  # (B, T)
        target_step_norms = torch.norm(target_action_seq, dim=-1)  # (B, T)
        
        # 计算平均步长（用于阈值判断）
        avg_step_norm = target_step_norms.mean()
        
        # 步长范数权重：鼓励预测大步长，惩罚预测小步长
        step_weights = torch.ones_like(target_step_norms)
        large_step_mask = target_step_norms > avg_step_norm
        small_step_mask = target_step_norms < avg_step_norm * 0.3  # 非常小的步长
        step_weights[large_step_mask] = 2.0  # 大步长给予2倍权重（重要的移动）
        # 对于小步长，我们给予小权重而不是大惩罚，避免网络专注于预测小值
        step_weights[small_step_mask] = 0.1  # 小步长给予较小权重
        
        # 计算每个时间步的基础损失
        l1_loss = F.l1_loss(predicted_action_seq, target_action_seq, reduction='none').mean(dim=-1)  # (B, T)
        mse_loss = F.mse_loss(predicted_action_seq, target_action_seq, reduction='none').mean(dim=-1)  # (B, T)
        
        # 应用步长权重
        weighted_l1_loss = l1_loss * step_weights
        weighted_mse_loss = mse_loss * step_weights
        
        # 方向一致性损失（角度损失：1 - cos(θ_pred, θ_target)）
        direction_loss = torch.tensor(0.0, device=predicted_action_seq.device)
        if predicted_action_seq.size(1) > 0:  # 至少有1个时间步
            # 计算预测动作和目标动作的方向向量（归一化）
            pred_norms = torch.norm(predicted_action_seq, dim=-1, keepdim=True)  # (B, T, 1)
            target_norms = torch.norm(target_action_seq, dim=-1, keepdim=True)  # (B, T, 1)
            
            # 创建有效掩码（排除零向量）  噪声滤波常量
            valid_mask = (pred_norms.squeeze(-1) > 1e-6) & (target_norms.squeeze(-1) > 1e-6)  # (B, T)
            
            if valid_mask.sum() > 0:
                # 归一化方向向量
                pred_directions = predicted_action_seq / (pred_norms + 1e-8)  # (B, T, 3)
                target_directions = target_action_seq / (target_norms + 1e-8)  # (B, T, 3)
                
                # 计算余弦相似度
                cosine_sim = (pred_directions * target_directions).sum(dim=-1)  # (B, T)
                cosine_sim = torch.clamp(cosine_sim, min=-1.0, max=1.0)  # 确保在[-1,1]范围内
                
                # 角度损失：1 - cos(θ)，只计算有效向量的损失
                angle_losses = 1.0 - cosine_sim  # (B, T)
                
                # 为角度损失添加步长权重：小步长的角度损失权重较小
                angle_weights = torch.ones_like(target_step_norms)
                small_angle_mask = target_step_norms < avg_step_norm * 0.5  # 小于平均步长的0.5倍
                angle_weights[small_angle_mask] = 0.3  # 小步长角度损失给予更小权重
                
                # 应用角度权重，只对有效掩码内的损失进行加权平均
                weighted_angle_losses = angle_losses * angle_weights
                direction_loss = weighted_angle_losses[valid_mask].mean()
        
        # 添加幅度损失：惩罚预测幅度与目标幅度差异过大
        magnitude_loss = F.mse_loss(predicted_step_norms, target_step_norms, reduction='mean')
        
        # 添加输出多样性损失：鼓励网络输出更多样化的增量，防止输出过于单一
        diversity_loss = torch.tensor(0.0, device=predicted_action_seq.device)
        if predicted_action_seq.size(0) > 1:  # 批次大小 > 1
            # 计算批次内预测的标准差，鼓励不同样本有不同的预测
            pred_std = torch.std(predicted_action_seq, dim=0).mean()  # (T, 3) -> scalar
            # 如果标准差太小，说明所有预测都很相似，给予惩罚
            # 使用 exp(-alpha * std) 保持值在 (0,1]，并显式限制到 [0,1] 以防数值异常
            diversity_loss = torch.exp(-pred_std * 10.0)
            diversity_loss = torch.clamp(diversity_loss, min=0.0, max=1.0)
        
        # 计算diversity_loss的分段衰减系数
        # 前50%回合：系数保持0.1不变
        # 后50%回合：系数从0.1线性衰减到0.01
        halfway_epoch = total_epochs // 2
        if current_epoch <= halfway_epoch:
            # 前50%回合，系数保持0.1
            diversity_weight = 0.1
        else:
            # 后50%回合，线性衰减 0.1 -> 0.01
            progress = (current_epoch - halfway_epoch) / (total_epochs - halfway_epoch)
            diversity_weight = 0.1 - (0.1 - 0.0) * progress  # 从0.1衰减
        
        # 计算总损失
        base_loss = 0.5 * weighted_l1_loss.mean() + 0.5 * weighted_mse_loss.mean()
        total_loss = 1.0 * base_loss + 1.0 * direction_loss + 0.0 * magnitude_loss + diversity_weight * diversity_loss
        
        
        # 计算标量损失用于记录
        l1_loss_scalar = l1_loss.mean()
        mse_loss_scalar = mse_loss.mean()
        
        # 评估指标
        with torch.no_grad():
            rmse_loss = torch.sqrt(mse_loss_scalar)
            step_norm_penalty = (step_weights - 1.0).mean()  # 平均步长惩罚
            
            # 计算真实损失（反归一化后的L1损失）
            real_l1_loss = 0.0
            real_l1_loss_max = 0.0
            if dataset is not None and hasattr(dataset, 'denormalize_data'):
                try:
                    # 反归一化预测值和目标值 (需要重塑为批次处理)
                    B, T, _ = predicted_action_seq.shape
                    pred_flat = predicted_action_seq.view(-1, 3).detach().cpu().numpy()
                    target_flat = target_action_seq.view(-1, 3).detach().cpu().numpy()
                    
                    pred_denorm = dataset.denormalize_data(pred_flat, 'actions')
                    target_denorm = dataset.denormalize_data(target_flat, 'actions')
                    
                    # 计算逐样本的真实L1损失
                    sample_real_l1_losses = np.mean(np.abs(pred_denorm - target_denorm), axis=1)  # (B*T,)
                    
                    # 计算平均值和最大值
                    real_l1_loss = np.mean(sample_real_l1_losses)
                    real_l1_loss_max = np.max(sample_real_l1_losses)
                except Exception as e:
                    print(f"⚠️  计算真实损失失败: {e}")
                    real_l1_loss = 0.0
                    real_l1_loss_max = 0.0
    else:
        # 原始模式：仅对最后时间步计算损失（增量预测）
        return TypeError("Invalid supervision mode")
        
    # 返回的是未作加权的损失值
    metrics = {
        'train_loss': total_loss.item(),
        'l1_error': l1_loss_scalar.item(),
        'mse_error': mse_loss_scalar.item(),
        'rmse_error': rmse_loss.item(),
        'direction_loss': direction_loss.item(),
        'magnitude_loss': magnitude_loss.item(),
        'diversity_loss': diversity_loss.item(),
        'diversity_weight': diversity_weight,  # 记录当前使用的diversity权重
        'step_norm_penalty': step_norm_penalty.item(),
        'avg_step_norm': avg_step_norm.item(),
        'real_l1_error(mm)': real_l1_loss * 1000,  # 真实损失（反归一化后）
        'real_l1_error_max(mm)': real_l1_loss_max * 1000,  # 每个batch中的最大真实损失
    }
    
    return total_loss, metrics


def prepare_feature_lstm_input_from_sequence_dataset(batch_data, dense_supervision=False):
    """
    从SequenceDataset批次中准备Feature-LSTM模型的输入 - 支持密集监督
    
    Args:
        batch_data: 来自SequenceDataset的批次数据
        dense_supervision: 是否使用密集监督
    
    Returns:
        dict: Feature-LSTM模型的输入字典
    """
    inputs = {
        'forces_l': batch_data['forces_l_seq'],      # (B, T, 3, 20, 20)
        'forces_r': batch_data['forces_r_seq'],      # (B, T, 3, 20, 20)
        'actions': batch_data['action_seq'],          # (B, T, 3)
        'seq_lengths': batch_data.get('seq_lengths', None)      # (B,) 可选
    }
    
    if dense_supervision:
        # 密集监督：提供所有时间步的目标动作序列
        inputs['target_action_seq'] = batch_data['target_action_seq']  # (B, T, 3)
    else:
        # 稀疏监督：只提供最后时间步的目标动作，以及当前动作用于增量预测
        inputs['target_next_action'] = batch_data['target_next_action']  # (B, 3)
        # 提供当前动作（序列的最后一个动作）用于增量预测
        inputs['current_action'] = batch_data['action_seq'][:, -1, :]  # (B, 3)
    
    return inputs


def create_tactile_policy_feature_lstm(config):
    """创建触觉策略Feature-LSTM模型"""
    return TactilePolicyFeatureLSTM(
        feature_dim=config.get('feature_dim', 128),
        action_dim=config.get('action_dim', 3),
        lstm_hidden_dim=config.get('lstm_hidden_dim', 256),
        lstm_num_layers=config.get('lstm_num_layers', 2),
        dropout_rate=config.get('dropout_rate', 0.25),
        pretrained_encoder_path=config.get('pretrained_encoder_path', None),
        action_embed_dim=config.get('action_embed_dim', 64),
        fc_hidden_dims=config.get('fc_hidden_dims', [128, 64])
    )


if __name__ == '__main__':
    # 简单测试
    config = {
        'feature_dim': 128,
        'action_dim': 3,
        'lstm_hidden_dim': 256,
        'lstm_num_layers': 2,
        'dropout_rate': 0.1,
        'pretrained_encoder_path': None,
        'action_embed_dim': 64,
        'fc_hidden_dims': [128, 64]
    }
    
    model = create_tactile_policy_feature_lstm(config)
    
    # 测试时序输入
    batch_size, seq_len = 4, 10
    forces_l_seq = torch.randn(batch_size, seq_len, 3, 20, 20)
    forces_r_seq = torch.randn(batch_size, seq_len, 3, 20, 20)
    action_seq = torch.randn(batch_size, seq_len, 3)
    
    output = model(forces_l_seq, forces_r_seq, action_seq)
    
    print(f"输入触觉力l序列形状: {forces_l_seq.shape}")
    print(f"输入触觉力r序列形状: {forces_r_seq.shape}")
    print(f"输入动作序列形状: {action_seq.shape}")
    print(f"输出下一动作形状: {output.shape}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试损失计算
    inputs = {'target_next_action': torch.randn_like(output)}
    loss, metrics = compute_feature_lstm_losses(inputs, output)
    
    print(f"总损失: {loss.item():.4f}")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
    
    print("✅ Feature-LSTM模型测试完成！")
