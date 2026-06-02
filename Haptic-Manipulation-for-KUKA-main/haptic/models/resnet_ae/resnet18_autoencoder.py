"""
ResNet18-based tactile autoencoder using torchvision (pretrained) encoder
Input: (B, 3, 20, 20)
Preprocess: optional rescale-to-[0,1] + ImageNet mean/std normalization
Loss: reuse your existing compute_cnn_autoencoder_losses()
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    import torchvision
    from torchvision.models import resnet18, ResNet18_Weights
except Exception as e:
    raise ImportError(f"torchvision is required: {e}")


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageNetPreprocess(nn.Module):
    """Optional: rescale to [0,1] then normalize with ImageNet stats.
    Args:
        rescale_to_01: if True, rescale per-sample globally to [0,1].
        clamp: whether to clamp after rescale.
        mean/std: tuples used for normalization (default: ImageNet stats).
    """
    def __init__(self, rescale_to_01: bool = True, clamp: bool = True,
                 mean=IMAGENET_MEAN, std=IMAGENET_STD):
        super().__init__()
        self.rescale_to_01 = rescale_to_01
        self.clamp = clamp
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4 and x.size(1) == 3, "expected (B,3,H,W)"
        y = x
        if self.rescale_to_01:
            # rescale per-sample using global min-max (across all channels)
            B = y.size(0)
            y_ = y.view(B, -1)
            minv = y_.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
            maxv = y_.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
            y = (y - minv) / (maxv - minv + 1e-6)
            if self.clamp:
                y = y.clamp(0.0, 1.0)
        # normalize per channel
        y = (y - self.mean) / self.std
        return y


class ResNet18Encoder(nn.Module):
    """torchvision resnet18 encoder with pretrained weights.
    Replaces the final fc to output latent_dim.
    """
    def __init__(self, latent_dim: int = 128, pretrained: bool = True, freeze_bn: bool = False,
                 keep_stem: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        # Optionally modify stem to better handle 20x20 (keep_stem=True keeps default)
        if not keep_stem:
            # Use stride=1 conv3x3, drop maxpool, to preserve resolution more
            self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.backbone.maxpool = nn.Identity()
        in_features = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Linear(in_features, latent_dim)
        if freeze_bn:
            self._freeze_batchnorm(self.backbone)

    @staticmethod
    def _freeze_batchnorm(module: nn.Module):
        for m in module.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class SimpleDecoder(nn.Module):
    """Decoder: latent -> (3,20,20) via fc + ConvTranspose2d stack"""
    def __init__(self, latent_dim: int = 128, out_channels: int = 3):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 5 * 5)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 5->10
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 10->20
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(x.size(0), 256, 5, 5)
        x = self.deconv(x)
        return x


class TactileResNet18Autoencoder(nn.Module):
    def __init__(self,
                 latent_dim: int = 128,
                 use_preprocess: bool = True,
                 rescale_to_01: bool = False,
                 pretrained: bool = True,
                 freeze_bn: bool = False,
                 keep_stem: bool = True):
        super().__init__()
        self.pre = ImageNetPreprocess(rescale_to_01=rescale_to_01) if use_preprocess else nn.Identity()
        self.encoder = ResNet18Encoder(latent_dim=latent_dim, pretrained=pretrained,
                                       freeze_bn=freeze_bn, keep_stem=keep_stem)
        self.decoder = SimpleDecoder(latent_dim=latent_dim, out_channels=3)

    def forward(self, x: torch.Tensor):
        x_n = self.pre(x)
        z = self.encoder(x_n)
        recon = self.decoder(z)
        return {"reconstructed": recon, "latent": z}

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.pre(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


def create_tactile_resnet18_autoencoder(config: Optional[dict] = None) -> nn.Module:
    config = config or {}
    return TactileResNet18Autoencoder(
        latent_dim=config.get('latent_dim', 128),
        use_preprocess=config.get('use_preprocess', True),
        rescale_to_01=config.get('rescale_to_01', True),
        pretrained=config.get('pretrained', True),
        freeze_bn=config.get('freeze_bn', False),
        keep_stem=config.get('keep_stem', True),
    )


if __name__ == "__main__":
    model = create_tactile_resnet18_autoencoder({"latent_dim": 128, "pretrained": True})
    x = torch.randn(2, 3, 20, 20)
    out = model(x)
    print(out['reconstructed'].shape, out['latent'].shape)
