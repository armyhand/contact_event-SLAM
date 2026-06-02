"""
Training script for ResNet18-based tactile autoencoder (torchvision pretrained encoder).
- Uses same wandb logging keys as your existing train_cnn_autoencoder.py
- Reuses compute_cnn_autoencoder_losses() for losses/metrics
- Works with your TactileForcesDataset pipeline
"""
import os
import sys
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

# Add project root to path
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
os.environ["WANDB_HTTP_TIMEOUT"] = "60"
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from DatasetTactile import create_train_test_tactile_datasets
from cnn_autoencoder import compute_cnn_autoencoder_losses
from resnet18_autoencoder import TactileResNet18Autoencoder

from ae_utils import save_comparison_images


def evaluate_model(model, dataloader, loss_config):
    """Evaluation loop returning averaged metrics dict."""
    model.eval()
    total_metrics = {}
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['image'].cuda(non_blocking=True)
            outputs = model(inputs)
            _, metrics = compute_cnn_autoencoder_losses(inputs, outputs, loss_config,
                                                        dataset=dataloader.dataset)
            bsz = inputs.size(0)
            total_samples += bsz
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v * bsz
    return {k: v/total_samples for k, v in total_metrics.items()}


def visualize_reconstruction(model, dataloader, output_dir, max_batches=None):
    model.eval()
    total_samples = len(dataloader.dataset)
    target_samples = max(1, total_samples // 400)
    if max_batches is None:
        batch_size = dataloader.batch_size
        max_batches = max(1, target_samples // batch_size)
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            inputs = batch['image'].cuda(non_blocking=True)
            outputs = model(inputs)
            recon = outputs['reconstructed']
            save_comparison_images(inputs.cpu(), recon.cpu(), output_dir,
                                   prefix=f"comparison_batch_{batch_idx}")


def train_resnet18_autoencoder(config):
    print("🚀 Start training ResNet18 AE...")

    # wandb login/init
    try:
        wandb.login()
        print("✅ wandb logged in")
    except Exception as e:
        print(f"⚠️  wandb login warning: {e}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(config['output']['output_dir'], f"resnet18_autoencoder_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    viz_dir = os.path.join(out_dir, "visualization")
    os.makedirs(viz_dir, exist_ok=True)

    run = wandb.init(
        project=config.get('wandb', {}).get('project', 'tactile-cnn-autoencoder'),
        name=config.get('wandb', {}).get('name', f"resnet18_run_{timestamp}"),
        config=config,
        dir=out_dir,
        tags=['resnet18-autoencoder', 'tactile', 'reconstruction', timestamp],
        notes='ResNet18 (torchvision pretrained) autoencoder for tactile reconstruction'
    )

    print("=" * 60)
    print("ResNet18 Autoencoder Training")
    print(f"Output Directory: {out_dir}")
    print(f"Data Root: {config['data']['data_root']}")
    print(f"Batch Size: {config['training']['batch_size']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Learning Rate: {config['training']['lr']}")
    print("=" * 60)

    # Datasets & loaders (unchanged)
    train_ds, test_ds, _ = create_train_test_tactile_datasets(
        data_root=config['data']['data_root'],
        categories=config['data']['categories'],
        start_frame=config['data']['start_frame'],
        normalize_method=config['data']['normalize_method']
    )
    train_loader = DataLoader(
        train_ds, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, 
        batch_size=config['training']['batch_size'], 
        shuffle=False, 
        pin_memory=True
    )
    # train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'],
    #                           shuffle=True, pin_memory=True)
    # test_loader = DataLoader(test_ds, batch_size=config['training']['batch_size'],
    #                          shuffle=False, pin_memory=True)

    # Model: torchvision resnet18 encoder + simple decoder
    model = TactileResNet18Autoencoder(
        latent_dim=config['model']['latent_dim'],
        pretrained=config['model'].get('pretrained', True),
        use_preprocess=config['model'].get('use_preprocess', True),
        rescale_to_01=config['model'].get('rescale_to_01', True),
        freeze_bn=config['model'].get('freeze_bn', False),
        keep_stem=config['model'].get('keep_stem', True),
    ).cuda()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")

    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'],
                                  weight_decay=config['training']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    best_loss = float('inf')

    for epoch in range(1, config['training']['epochs'] + 1):
        model.train()
        total_metrics = {}
        total_samples = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{config['training']['epochs']}"):
            inputs = batch['image'].cuda(non_blocking=True)
            outputs = model(inputs)
            loss, metrics = compute_cnn_autoencoder_losses(inputs, outputs, config['loss'],
                                                           dataset=train_ds)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bsz = inputs.size(0)
            total_samples += bsz
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v * bsz

        # Average metrics
        avg_metrics = {k: v / total_samples for k, v in total_metrics.items()}
        avg_loss = avg_metrics['total_loss']

        # LR scheduling
        scheduler.step(avg_loss)
        cur_lr = optimizer.param_groups[0]['lr']

        # wandb log (same keys as your original)
        log_dict = {'train/learning_rate': cur_lr}
        for k, v in avg_metrics.items():
            log_dict[f'train/{k}'] = v
        run.log(log_dict, step=epoch)

        print(f"Epoch {epoch}/{config['training']['epochs']}")
        print(f"  Learning Rate: {cur_lr:.6e}")
        for k, v in avg_metrics.items():
            print(f"  {k}: {v:.6f}")
        print("-" * 50)

        # Validation
        if epoch % config['training']['eval_every'] == 0:
            val_metrics = evaluate_model(model, test_loader, config['loss'])
            run.log({f'val/{k}': v for k, v in val_metrics.items()}, step=epoch)

            print("Validation:")
            for k, v in val_metrics.items():
                print(f"  val_{k}: {v:.6f}")
            print("-" * 50)

        # Visualization every 10 epochs (optional)
        if epoch % 10 == 0:
            epoch_viz_dir = os.path.join(viz_dir, f"epoch_{epoch}")
            os.makedirs(epoch_viz_dir, exist_ok=True)
            visualize_reconstruction(model, train_loader, epoch_viz_dir)

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(out_dir, "best_model.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
                'config': config
            }, best_path)
            wandb.save(best_path)
            print(f"💾 Saved best model (Loss: {best_loss:.6f})")

    # Final save
    final_path = os.path.join(out_dir, "final_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': avg_loss,
        'config': config
    }, final_path)
    wandb.save(final_path)

    print("✅ Training complete!")
    return model, best_loss


def main(config):
    return train_resnet18_autoencoder(config)


if __name__ == '__main__':
    # Default config: keep same structure/keys as your original for wandb parity
    config = {
        'data': {
            'data_root': 'data25.7_aligned',
            'categories': [
                'cir_lar', 'cir_med', 'cir_sma',
                'rect_lar', 'rect_med', 'rect_sma',
                'tri_lar', 'tri_med', 'tri_sma'
            ],
            'start_frame': 0,
            # If you want ImageNet normalization inside the model, set this to None to avoid double-normalization
            'normalize_method': 'zscore'  
        },
        'wandb': {
            'project': 'tactile-latent-autoencoder',
            'name': 'resnet18_ae_run'
        },
        'model': {
            'in_channels': 3,              # kept for parity; not used directly
            'latent_dim': 128,
            'pretrained': True,
            'use_preprocess': True,        # enable ImageNet mean/std normalization in-model
            'rescale_to_01': False,         # rescale each sample to [0,1] before normalization
            'freeze_bn': False,
            'keep_stem': True              # set False to use 3x3 s=1 stem (better for 20x20)
        },
        'loss': {
            'l2_lambda': 0.001,
            'use_resultant_loss': False,
            'force_lambda': 1e-5,
            'moment_lambda': 5e-7
        },
        'training': {
            'batch_size': 32,
            'epochs': 100,
            'lr': 1e-4,
            'weight_decay': 1e-4,
            'eval_every': 1
        },
        'output': {
            'output_dir': 'ae_checkpoints'
        }
    }
    main(config)
