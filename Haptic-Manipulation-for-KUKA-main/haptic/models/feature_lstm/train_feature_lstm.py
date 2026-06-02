"""
Feature-LSTM策略模型训练脚本 - 时序版本
输入: 时序触觉数据 forces_l[seq_len, 3, 20, 20] + forces_r[seq_len, 3, 20, 20] + action_seq[seq_len, 3]
输出: action_nextstep[3]
"""
import os
import sys
import torch
import wandb
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

# 设置代理（如果需要代理才能访问外网）
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"
os.environ["WANDB_HTTP_TIMEOUT"] = "60"

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# 项目根路径
project_root = os.path.abspath(os.path.dirname(__file__))

from DatasetSequence import create_sequence_datasets
from feature_lstm import create_tactile_policy_feature_lstm, compute_feature_lstm_losses, prepare_feature_lstm_input_from_sequence_dataset


def train_feature_lstm_policy(config):
    """
    训练Feature-LSTM策略模型
    """
    print("🚀 开始Feature-LSTM策略训练...")
    
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config['output']['output_dir'], f"feature_lstm_policy_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化 wandb
    run = wandb.init(
        project=config.get('wandb', {}).get('project', 'tactile-action-learn'),
        name=config.get('wandb', {}).get('name'),
        config=config,
        dir=output_dir,
        tags=['feature-lstm-policy', 'sequence-prediction'] + [timestamp],
        notes='Feature-LSTM policy training with sequential tactile data'
    )
    
    # 设置设备
    torch.cuda.set_device(1)
    device = torch.device('cuda:1')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 使用设备: {device}")
    
    print("=" * 60)
    print("Feature-LSTM Policy Training")
    print(f"Output Directory: {output_dir}")
    print(f"Data Root: {config['data']['data_root']}")
    print(f"Batch Size: {config['training']['batch_size']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Learning Rate: {config['training']['lr']}")
    print(f"Sequence Length: {config['data']['sequence_length']}")
    print(f"Sampling Interval: {config['data']['sampling_interval']}")
    print("=" * 60)
    print("Model Configuration:")
    print(config['model'])
    print("=" * 60)
    
    # 创建时序数据集
    print("📂 加载时序数据集...")
    train_dataset, test_dataset, normalization_params = create_sequence_datasets(
        data_root=config['data']['data_root'],
        categories=config['data']['categories'],
        sequence_length=config['data']['sequence_length'],
        prediction_step=config['data']['sampling_interval'],
        overlap_stride=config['data'].get('overlap_stride', 5),
        use_forces=True,  # Feature-LSTM需要触觉数据
        normalization_config=config['data'].get('normalization_config', None)
    )
    
    # 将计算出的归一化参数更新到config中
    if normalization_params:
        # 更新config中的normalization_config为计算出的实际参数
        config['data']['normalization_config'] = normalization_params
        print("📊 归一化参数已更新到config中")
        
        # 打印参数信息
        for data_type, params in normalization_params.items():
            if 'params' in params and params['params']:
                print(f"   {data_type}: {params['method']}")
                if data_type == 'actions' and isinstance(params['params'], dict):
                    # 逐轴参数
                    if any(key.startswith('axis_') for key in params['params'].keys()):
                        for axis_name, axis_params in params['params'].items():
                            if isinstance(axis_params, dict):
                                print(f"     {axis_name}: mean={axis_params.get('mean', 'N/A'):.4f}, std={axis_params.get('std', 'N/A'):.4f}")
                    else:
                        # 全局参数
                        print(f"     global: mean={params['params'].get('mean', 'N/A'):.4f}, std={params['params'].get('std', 'N/A'):.4f}")
                else:
                    # 其他数据类型的全局参数
                    print(f"     mean={params['params'].get('mean', 'N/A'):.4f}, std={params['params'].get('std', 'N/A'):.4f}")
        
        # 直接更新WandB的config，这样会写入到config.yaml文件中
        print("📝 正在将归一化参数写入WandB config.yaml文件...")
        
        # 方法1：更新完整的config结构到WandB（允许值变更）
        wandb.config.update(config, allow_val_change=True)
        
        # 方法2：单独更新归一化参数（确保被记录）
        wandb.config.update({
            'computed_normalization_params': normalization_params,
            'normalization_computed_at_runtime': True
        }, allow_val_change=True)
        
        print("✅ 归一化参数已写入WandB config.yaml文件")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True if device.type == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True if device.type == 'cuda' else False
    )

    print(f"✅ 训练集: {len(train_dataset)} 序列")
    print(f"✅ 测试集: {len(test_dataset)} 序列")

    # 创建模型
    print("🏗️ 创建Feature-LSTM模型...")
    model = create_tactile_policy_feature_lstm(config['model']).to(device)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 优化器和调度器
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['training']['lr'], 
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # 训练循环
    best_loss = float('inf')
    best_model_path = None
    
    # 密集监督配置
    dense_supervision = config['training'].get('dense_supervision', False)
    print(f"🎯 监督模式: {'密集监督 (每时间步)' if dense_supervision else '稀疏监督 (最后一步)'}")
    
    try:
        for epoch in range(config['training']['epochs']):
            print(f"\n🔄 Epoch {epoch + 1}/{config['training']['epochs']}")
            
            # 训练阶段
            train_loss, train_metrics = train_epoch(model, train_loader, optimizer, device, train_dataset, dense_supervision=dense_supervision, current_epoch=epoch+1, total_epochs=config['training']['epochs'])
            
            print(f"📈 训练结果:")
            print(f"   Loss: {train_loss:.6f}")
            print(f"   L1 error: {train_metrics.get('l1_error', 0):.6f}")
            print(f"   mse error: {train_metrics.get('mse_error', 0):.6f}")
            print(f"   direction_loss: {train_metrics.get('direction_loss', 0):.6f}")
            print(f"   magnitude_loss: {train_metrics.get('magnitude_loss', 0):.6f}")
            print(f"   diversity_loss: {train_metrics.get('diversity_loss', 0):.6f}")
            print(f"   diversity_weight: {train_metrics.get('diversity_weight', 0):.6f}")
            print(f"   step_norm_penalty: {train_metrics.get('step_norm_penalty', 0):.6f}")
            print(f"   avg_step_norm: {train_metrics.get('avg_step_norm', 0):.6f}")
            print(f"   Real L1 Error: {train_metrics.get('real_l1_error(mm)', 0):.2f} mm")
            print(f"   Real L1 Max: {train_metrics.get('real_l1_error_max(mm)', 0):.2f} mm")
            
            # 记录训练指标到 wandb
            train_wandb_log = {'learning_rate': optimizer.param_groups[0]['lr']}
            for key, value in train_metrics.items():
                train_wandb_log[f'train/{key}'] = value
            
            run.log(train_wandb_log, step=epoch)

            # 验证阶段
            if (epoch + 1) % config['training'].get('eval_every', 1) == 0:
                test_loss, test_metrics = evaluate(model, test_loader, device, test_dataset, dense_supervision=dense_supervision, current_epoch=epoch+1, total_epochs=config['training']['epochs'])
                
                print(f"📊 验证结果:")
                print(f"   Loss: {test_loss:.6f}")
                print(f"   l1_error: {test_metrics.get('l1_error', 0):.6f}")
                print(f"   mse_error: {test_metrics.get('mse_error', 0):.6f}")
                print(f"   direction_loss: {test_metrics.get('direction_loss', 0):.6f}")
                print(f"   magnitude_loss: {test_metrics.get('magnitude_loss', 0):.6f}")
                print(f"   diversity_loss: {test_metrics.get('diversity_loss', 0):.6f}")
                print(f"   diversity_weight: {test_metrics.get('diversity_weight', 0):.6f}")
                print(f"   step_norm_penalty: {test_metrics.get('step_norm_penalty', 0):.6f}")
                print(f"   avg_step_norm: {test_metrics.get('avg_step_norm', 0):.6f}")
                print(f"   Real L1 Error: {test_metrics.get('real_l1_error(mm)', 0):.2f} mm")
                print(f"   Real L1 Max: {test_metrics.get('real_l1_error_max(mm)', 0):.2f} mm")

                
                # 记录验证指标到 wandb
                val_wandb_log = {}
                for key, value in test_metrics.items():
                    val_wandb_log[f'val/{key}'] = value
                
                run.log(val_wandb_log, step=epoch)
                
                # 保存最佳模型
                if test_loss < best_loss:
                    best_loss = test_loss
                    best_model_path = os.path.join(output_dir, "best_model.pt")
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'test_loss': test_loss,
                        'config': config,
                        'normalization_params': normalization_params
                    }, best_model_path)
                    print(f"💾 保存最佳模型: {best_model_path}")
                
                # 学习率调度
                scheduler.step(test_loss)
            
            # 定期保存检查点
            if (epoch + 1) % config['training'].get('save_every', 20) == 0:
                checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pt")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'test_loss': test_loss if 'test_loss' in locals() else None,
                    'config': config,
                    'normalization_params': normalization_params
                }, checkpoint_path)
                print(f"💾 保存检查点: {checkpoint_path}")
        
        # 保存最终模型
        final_model_path = os.path.join(output_dir, "final_model.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'train_loss': train_loss,
            'test_loss': test_loss if 'test_loss' in locals() else None,
            'config': config,
            'normalization_params': normalization_params
        }, final_model_path)
        
        # 保存到 wandb
        wandb.save(final_model_path)
        if best_model_path:
            wandb.save(best_model_path)
        
        print("✅ Feature-LSTM策略模型训练完成!")
        print(f"📈 最佳验证损失: {best_loss:.6f}")
        
        return model, best_loss, best_model_path
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        raise
    finally:
        run.finish()


def train_epoch(model, train_loader, optimizer, device, dataset=None, dense_supervision=False, current_epoch=1, total_epochs=10):
    """训练一个epoch - 适配简单版本的feature_lstm"""
    model.train()
    total_loss = 0.0
    total_metrics = {}
    total_samples = 0
    
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
        # 准备输入数据
        lstm_inputs = prepare_feature_lstm_input_from_sequence_dataset(batch, dense_supervision=dense_supervision)
        for key in lstm_inputs:
            if isinstance(lstm_inputs[key], torch.Tensor):
                lstm_inputs[key] = lstm_inputs[key].to(device)
        
        # 前向传播
        optimizer.zero_grad()
        forces_l = lstm_inputs['forces_l']  # (B, T, 3, 20, 20)
        forces_r = lstm_inputs['forces_r']  # (B, T, 3, 20, 20)
        actions = lstm_inputs['actions']    # (B, T, 3)
        seq_lengths = lstm_inputs.get('seq_lengths', None)  # (B,) 可选
        
        # 根据是否使用密集监督决定模型输出
        outputs = model(forces_l, forces_r, actions, seq_lengths, return_all_steps=dense_supervision)
        
        # 计算损失（使用简单版本的损失函数）
        loss, metrics = compute_feature_lstm_losses(
            lstm_inputs, outputs, dataset=dataset, 
            dense_supervision=dense_supervision,
            current_epoch=current_epoch,
            total_epochs=total_epochs
        )
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 累积统计
        batch_size = forces_l.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        # 累积指标
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if key not in total_metrics:
                    total_metrics[key] = 0
                total_metrics[key] += value * batch_size
    
    avg_loss = total_loss / max(total_samples, 1)
    avg_metrics = {key: value / max(total_samples, 1) 
                   for key, value in total_metrics.items()}
    
    return avg_loss, avg_metrics


def evaluate(model, test_loader, device, dataset=None, dense_supervision=False, current_epoch=1, total_epochs=10):
    """评估模型 - 支持密集监督"""
    model.eval()
    total_loss = 0.0
    total_metrics = {}
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # 准备输入数据
            lstm_inputs = prepare_feature_lstm_input_from_sequence_dataset(batch, dense_supervision=dense_supervision)
            for key in lstm_inputs:
                if isinstance(lstm_inputs[key], torch.Tensor):
                    lstm_inputs[key] = lstm_inputs[key].to(device)
            
            # 前向传播
            forces_l = lstm_inputs['forces_l']  # (B, T, 3, 20, 20)
            forces_r = lstm_inputs['forces_r']  # (B, T, 3, 20, 20)
            actions = lstm_inputs['actions']    # (B, T, 3)
            seq_lengths = lstm_inputs.get('seq_lengths', None)  # (B,) 可选
            
            # 根据是否使用密集监督决定模型输出
            outputs = model(forces_l, forces_r, actions, seq_lengths, return_all_steps=dense_supervision)
            
            # 计算损失（使用简单版本的损失函数）
            loss, metrics = compute_feature_lstm_losses(
                lstm_inputs, outputs, dataset=dataset, 
                dense_supervision=dense_supervision,
                current_epoch=current_epoch,
                total_epochs=total_epochs
            )
            
            # 累加统计
            batch_size = forces_l.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # 累加指标
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if key not in total_metrics:
                        total_metrics[key] = 0.0
                    total_metrics[key] += value * batch_size
    
    avg_loss = total_loss / max(total_samples, 1)
    avg_metrics = {key: value / max(total_samples, 1) 
                   for key, value in total_metrics.items()}
    
    return avg_loss, avg_metrics


def main(config):
    """主函数"""
    print("🎯 Feature-LSTM策略训练开始")
    print(f"📊 配置摘要:")
    print(f"  - 序列长度: {config['data']['sequence_length']}")
    print(f"  - 采样间隔: {config['data']['sampling_interval']}")
    print(f"  - 批次大小: {config['training']['batch_size']}")
    print(f"  - 训练轮数: {config['training']['epochs']}")
    print(f"  - 学习率: {config['training']['lr']}")
    
    train_feature_lstm_policy(config)


if __name__ == '__main__':
    # 默认配置
    config = {
        'data': {
            'data_root': 'data25.7_aligned',
            'categories': [
                "cir_lar", "cir_med", "cir_sma",
                "rect_lar", "rect_med", "rect_sma", 
                "tri_lar", "tri_med", "tri_sma",
            ],
            'sequence_length': 10,      # LSTM输入序列长度
            'sampling_interval': 5,    # 采样间隔T: [0, T, 2T, 3T, 4T] -> 5T
            'overlap_stride': 1,       # 序列重叠步长
            'normalization_config': {
                'actions': {'method': 'zscore', 'params': None, 'axis_mode': 'global'},
                'forces': {'method': 'zscore', 'params': None}
            }
        },
        'model': {
            'feature_dim': 128,              # CNN特征维度
            'action_dim': 3,                 # 输出动作维度
            'lstm_hidden_dim': 256,          # LSTM隐藏维度
            'lstm_num_layers': 2,            # LSTM层数
            'dropout_rate': 0.2,             # Dropout比率
            'pretrained_encoder_path': 'cnnae_crt_128.pt',  # 预训练CNN编码器路径
            'action_embed_dim': 64,          # 动作嵌入维度
            'fc_hidden_dims': [256, 128, 64] # 全连接层维度
        },
        'training': {
            'batch_size': 24,        # 时序数据较大，使用较小批次
            'epochs': 15,           
            'lr': 5e-4,              # 较小学习率用于微调
            'weight_decay': 1e-5,
            'eval_every': 1,         # 每 eval_every 轮验证
            'save_every': 10,        # 每 save_every 轮保存一次检查点
            'dense_supervision': True,  # 启用密集监督 (每个时间步都监督)
        },
        'wandb': {
            'project': "tactile-action-learn-test",
            'name': 'feature-lstm-delta_policy-AIDO-Div0.1decay0-LargeStep',  # absolute input delta output
        },
        'output': {
            'output_dir': 'checkpoints'
        }
    }
    
    # 检查路径
    data_path = os.path.join(project_root, config['data']['data_root'])
    config['data']['data_root'] = data_path
    config['output']['output_dir'] = os.path.join(project_root, config['output']['output_dir'])
    
    # 检查预训练模型路径
    pretrained_path = config['model']['pretrained_encoder_path']
    if pretrained_path and not os.path.isabs(pretrained_path):
        pretrained_full_path = os.path.join(project_root, pretrained_path)
        if os.path.exists(pretrained_full_path):
            config['model']['pretrained_encoder_path'] = pretrained_full_path
            print(f"✅ 预训练模型路径: {pretrained_full_path}")
        else:
            print(f"⚠️  预训练模型不存在，将使用随机初始化: {pretrained_full_path}")
            config['model']['pretrained_encoder_path'] = None
    
    if os.path.exists(data_path):
        print(f"✅ 数据路径存在: {data_path}")
        main(config)
    else:
        print(f"❌ 数据路径不存在: {data_path}")