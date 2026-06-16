"""GAN 模型结构定义。

本文件集中定义项目中用到的神经网络模块：
DCGAN 生成器/判别器、CycleGAN 生成器/PatchGAN 判别器，以及轻量
StyleGAN 风格生成器。训练脚本只负责组织训练流程，模型细节统一放在这里。
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm


def maybe_sn(module: nn.Conv2d, norm: str) -> nn.Module:
    if norm in ["sn", "sngp"]:
        return spectral_norm(module)
    else:
        return module
    
def norm_layer(ngf, norm: str) -> nn.Module:
    if norm in ["sn", "wo", "sngp"]:
        return nn.Identity()
    elif norm == "gp":
        return nn.InstanceNorm2d(ngf, affine=True)
    elif norm == "bn":
        return nn.BatchNorm2d(ngf)


class Generator(nn.Module):
    """DCGAN 生成器：把随机噪声逐步上采样为 64x64 RGB 头像。"""

    def __init__(self, latent_dim: int = 100, image_channels: int = 3, feature_maps: int = 64) -> None:
        super().__init__()
        ngf = feature_maps

        self.net = nn.Sequential(
            # 输入: N x latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # N x (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # N x (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # N x (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # N x ngf x 32 x 32
            nn.ConvTranspose2d(ngf, image_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
            # 输出: N x 3 x 64 x 64，像素范围 [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """把形状为 [N, latent_dim] 或 [N, latent_dim, 1, 1] 的噪声映射为图片。"""

        if z.ndim == 2:
            z = z[:, :, None, None]
        return self.net(z)


class Discriminator(nn.Module):
    """DCGAN 判别器：判断输入头像来自真实数据还是生成器。"""

    def __init__(self, image_channels: int = 3, feature_maps: int = 64, norm: str="bn") -> None:
        super().__init__()
        ndf = feature_maps
        if norm == "sn":
            bias_enable = True
        else: 
            bias_enable = False
        self.net = nn.Sequential(
            # 输入: N x 3 x 64 x 64
            maybe_sn(nn.Conv2d(image_channels, ndf, kernel_size=4, stride=2, padding=1, bias=bias_enable), norm=norm),
            nn.LeakyReLU(0.2, inplace=True),
            # N x ndf x 32 x 32
            maybe_sn(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=bias_enable), norm=norm),
            norm_layer(ndf * 2, norm),
            nn.LeakyReLU(0.2, inplace=True),
            # N x (ndf*2) x 16 x 16
            maybe_sn(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=bias_enable), norm=norm),
            norm_layer(ndf * 4, norm),
            nn.LeakyReLU(0.2, inplace=True),
            # N x (ndf*4) x 8 x 8
            maybe_sn(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=bias_enable), norm=norm),
            norm_layer(ndf * 8, norm),
            nn.LeakyReLU(0.2, inplace=True),
            # N x (ndf*8) x 4 x 4
            maybe_sn(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=bias_enable), norm=norm),
            # 输出真假 logits，训练时配合 BCEWithLogitsLoss，兼容 mixed precision。
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """输入图片张量，输出每张图片对应的真假 logits。"""

        return self.net(image).view(-1)


def init_dcgan_weights(module: nn.Module) -> None:
    """DCGAN 论文推荐的初始化方式。"""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        weight = module.weight_orig if hasattr(module, "weight_orig") else module.weight
        nn.init.normal_(weight.data, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias.data)
    elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
        if module.weight is not None:
            nn.init.normal_(module.weight.data, 1.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias.data)


