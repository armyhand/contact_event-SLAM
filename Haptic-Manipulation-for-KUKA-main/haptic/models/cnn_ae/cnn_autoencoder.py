"""
CNN自编码器模型 - 用于触觉力数据重建
输入形状: (3, 20, 20) - 三通道触觉力图像
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        return F.relu(out)


class TactileCNNEncoder(nn.Module):
    """触觉CNN编码器"""
    def __init__(self, in_channels=3, latent_dim=128):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # 初始卷积 (3, 20, 20) -> (64, 20, 20)
            nn.Conv2d(in_channels, 64, 7, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 第一个残差块组 (64, 20, 20) -> (64, 10, 10)
            ResidualBlock(64, 64),
            ResidualBlock(64, 64, stride=2),
            
            # 第二个残差块组 (64, 10, 10) -> (128, 5, 5)
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
            
            # 第三个残差块组 (128, 5, 5) -> (256, 3, 3)
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256),
            
            # 自适应池化到固定大小
            nn.AdaptiveAvgPool2d((5, 5)),
        )
        
        # 全连接层映射到潜在空间
        self.fc = nn.Linear(256 * 5 * 5, latent_dim)
        
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TactileCNNDecoder(nn.Module):
    """触觉CNN解码器"""
    def __init__(self, latent_dim=128, out_channels=3):
        super().__init__()
        
        # 从潜在向量到特征图
        self.fc = nn.Linear(latent_dim, 256 * 5 * 5)
        
        self.decoder = nn.Sequential(
            # 从 (256, 5, 5) 上采样到 (128, 10, 10)
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 从 (128, 10, 10) 上采样到 (64, 20, 20)
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 输出层 (64, 20, 20) -> (3, 20, 20)
            nn.Conv2d(64, out_channels, 3, padding=1),
        )
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 256, 5, 5)
        x = self.decoder(x)
        return x


class TactileCNNAutoencoder(nn.Module):
    """触觉CNN自编码器"""
    def __init__(self, in_channels=3, latent_dim=128):
        super().__init__()
        self.encoder = TactileCNNEncoder(in_channels, latent_dim)
        self.decoder = TactileCNNDecoder(latent_dim, in_channels)
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return {
            'reconstructed': reconstructed,
            'latent': latent
        }
    
    def encode(self, x):
        """编码输入到潜在空间"""
        return self.encoder(x)
    
    def decode(self, z):
        """从潜在空间解码"""
        return self.decoder(z)


def compute_resultant_force_and_moment(force_maps):
    """
    计算触觉力图的合力和合力矩
    
    Args:
        force_maps: (B, 3, H, W) 触觉力图，3个通道分别是X, Y, Z方向的力
    
    Returns:
        resultant_force: (B, 3) 合力 [Fx, Fy, Fz]
        resultant_moment: (B, 3) 合力矩 [Mx, My, Mz]
    """
    B, C, H, W = force_maps.shape
    
    # 提取XYZ方向的力
    fx = force_maps[:, 0, :, :]  # (B, H, W)
    fy = force_maps[:, 1, :, :]  # (B, H, W)
    fz = force_maps[:, 2, :, :]  # (B, H, W)
    
    # 计算合力：对所有像素求和
    resultant_fx = torch.sum(fx, dim=(1, 2))  # (B,)
    resultant_fy = torch.sum(fy, dim=(1, 2))  # (B,)
    resultant_fz = torch.sum(fz, dim=(1, 2))  # (B,)
    
    resultant_force = torch.stack([resultant_fx, resultant_fy, resultant_fz], dim=1)  # (B, 3)
    
    # 创建位置网格 (像素坐标系)
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=force_maps.device),
        torch.arange(W, dtype=torch.float32, device=force_maps.device),
        indexing='ij'
    )
    
    # 将坐标中心化到传感器中心
    center_x, center_y = W / 2.0, H / 2.0
    x_coords = x_coords - center_x  # (H, W)
    y_coords = y_coords - center_y  # (H, W)
    
    # 扩展到批次维度
    x_coords = x_coords.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)
    y_coords = y_coords.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)
    
    # 计算力矩：M = r × F
    # Mx = y*Fz - z*Fy (这里z=0，因为是2D传感器表面)
    # My = z*Fx - x*Fz (这里z=0)  
    # Mz = x*Fy - y*Fx
    mx = y_coords * fz  # y*Fz，忽略z*Fy项（z=0）
    my = -x_coords * fz  # -x*Fz，忽略z*Fx项（z=0）
    mz = x_coords * fy - y_coords * fx  # x*Fy - y*Fx
    
    # 对所有像素求和得到总力矩
    resultant_mx = torch.sum(mx, dim=(1, 2))  # (B,)
    resultant_my = torch.sum(my, dim=(1, 2))  # (B,)
    resultant_mz = torch.sum(mz, dim=(1, 2))  # (B,)
    
    resultant_moment = torch.stack([resultant_mx, resultant_my, resultant_mz], dim=1)  # (B, 3)
    
    return resultant_force, resultant_moment


def compute_cnn_autoencoder_losses(inputs, outputs, config, dataset=None):
    """
    计算CNN自编码器损失
    
    Args:
        inputs: 输入数据 (B, 3, 20, 20)
        outputs: 模型输出字典
        config: 损失配置
        dataset: 数据集对象，用于反归一化计算真实物理损失
    
    Returns:
        loss: 总损失
        metrics: 损失分解字典
    """
    reconstructed = outputs['reconstructed']
    latent = outputs['latent']
    
    # 1. 重建损失 (像素级MSE)
    recon_loss = F.mse_loss(reconstructed, inputs)
    
    # 2. L2正则化损失（主要更新De，规范latent）
    l2_loss = torch.norm(latent, p=2, dim=1).mean()
    
    # 3. 合力和合力矩损失
    force_loss = torch.tensor(0.0, device=inputs.device)
    moment_loss = torch.tensor(0.0, device=inputs.device)
    real_force_loss = [0.0, 0.0, 0.0, 0.0]  # [fx, fy, fz, total]
    real_moment_loss = [0.0, 0.0, 0.0, 0.0]  # [mx, my, mz, total]
    
    # 计算原始和重建的合力、合力矩
    orig_force, orig_moment = compute_resultant_force_and_moment(inputs)
    recon_force, recon_moment = compute_resultant_force_and_moment(reconstructed)
    
    # 归一化空间的损失（用于梯度计算，归一化后的像素损失）
    force_loss = F.mse_loss(recon_force, orig_force)
    moment_loss = F.mse_loss(recon_moment, orig_moment)
    
    # 计算真实物理单位的损失（用于监控）
    if dataset is not None and hasattr(dataset, 'denormalize_data'):
        try:
            # 将触觉数据反归一化到真实物理单位
            inputs_denorm = dataset.denormalize_data(inputs.detach().cpu().numpy())
            reconstructed_denorm = dataset.denormalize_data(reconstructed.detach().cpu().numpy())
            
            # 转回tensor并计算真实物理单位的合力和合力矩
            inputs_denorm_tensor = torch.from_numpy(inputs_denorm).to(inputs.device)
            reconstructed_denorm_tensor = torch.from_numpy(reconstructed_denorm).to(inputs.device)
            
            orig_force_real, orig_moment_real = compute_resultant_force_and_moment(inputs_denorm_tensor)
            recon_force_real, recon_moment_real = compute_resultant_force_and_moment(reconstructed_denorm_tensor)
            
            # 计算各轴向的MSE损失
            force_diff = recon_force_real - orig_force_real  # (B, 3)
            moment_diff = recon_moment_real - orig_moment_real  # (B, 3)
            
            # 计算三轴分量的MSE
            force_mse_per_axis = torch.mean(force_diff ** 2, dim=0)  # (3,) [fx, fy, fz]
            moment_mse_per_axis = torch.mean(moment_diff ** 2, dim=0)  # (3,) [mx, my, mz]
            
            # 计算总体MSE
            force_mse_total = torch.mean(force_diff ** 2)  # 标量
            moment_mse_total = torch.mean(moment_diff ** 2)  # 标量
            
            # 转换为列表格式 [x, y, z, total]
            real_force_loss = [
                force_mse_per_axis[0].item(),  # fx
                force_mse_per_axis[1].item(),  # fy
                force_mse_per_axis[2].item(),  # fz
                force_mse_total.item()         # total
            ]
            
            real_moment_loss = [
                moment_mse_per_axis[0].item(),  # mx
                moment_mse_per_axis[1].item(),  # my
                moment_mse_per_axis[2].item(),  # mz
                moment_mse_total.item()         # total
            ]
            
        except Exception as e:
            print(f"⚠️  计算真实物理损失失败: {e}")
            real_force_loss = [0.0, 0.0, 0.0, 0.0]
            real_moment_loss = [0.0, 0.0, 0.0, 0.0]
        
    # 4. 总损失
    if config.get('use_resultant_loss', False):
        total_loss = (recon_loss + 
                config.get('l2_lambda', 0.001) * l2_loss +
                config.get('force_lambda', 1e-5) * force_loss +
                config.get('moment_lambda', 5e-7) * moment_loss)
    else:
        total_loss = (recon_loss + 
                      config.get('l2_lambda', 0.001) * l2_loss)
    
    metrics = {
        'recon_loss(mse)': recon_loss.item(),
        'l2_loss': l2_loss.item(),
        'force_loss': force_loss.item(),
        'moment_loss': moment_loss.item(),
        'real_force_loss_x(N)': real_force_loss[0],  # X轴合力损失
        'real_force_loss_y(N)': real_force_loss[1],  # Y轴合力损失
        'real_force_loss_z(N)': real_force_loss[2],  # Z轴合力损失
        'real_force_loss_total(N)': real_force_loss[3],  # 总合力损失
        'real_moment_loss_x(N*pixel)': real_moment_loss[0],  # X轴力矩损失
        'real_moment_loss_y(N*pixel)': real_moment_loss[1],  # Y轴力矩损失
        'real_moment_loss_z(N*pixel)': real_moment_loss[2],  # Z轴力矩损失
        'real_moment_loss_total(N*pixel)': real_moment_loss[3],  # 总力矩损失
        'total_loss': total_loss.item()
    }
    
    return total_loss, metrics


# 便利函数
def create_tactile_cnn_autoencoder(config):
    """创建触觉CNN自编码器"""
    return TactileCNNAutoencoder(
        in_channels=config.get('in_channels', 3),
        latent_dim=config.get('latent_dim', 128)
    )


if __name__ == '__main__':
    # 测试模型
    model = TactileCNNAutoencoder(latent_dim=128)
    
    # 测试输入
    x = torch.randn(4, 3, 20, 20)
    
    # 前向传播
    outputs = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"重建形状: {outputs['reconstructed'].shape}")
    print(f"潜在向量形状: {outputs['latent'].shape}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
