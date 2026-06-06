"""GAN 模型结构定义。

本文件集中定义项目中用到的神经网络模块：
DCGAN 生成器/判别器、CycleGAN 生成器/PatchGAN 判别器，以及轻量
StyleGAN 风格生成器。训练脚本只负责组织训练流程，模型细节统一放在这里。
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm



def norm_act(dim, use_spec_norm: bool):
    if not use_spec_norm:
        return nn.Sequential(nn.BatchNorm2d(dim), nn.LeakyReLU(0.2, inplace=True))
    else:
        return nn.LeakyReLU(0.2, inplace=True)



        
class ScaledSpectralNormConv2d(nn.Module):
    """根据 ABCAS 论文 Fig. 1 设定的动态缩放谱归一化层。"""
    def __init__(self, conv_layer: nn.Conv2d, scale_provider) -> None:
        super().__init__()
        self.conv = spectral_norm(conv_layer)
        self.scale_provider = scale_provider

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 获取由训练循环动态计算并注入的 m 值
        m = self.scale_provider()
        import random
        if random.random() < 0.01:  # 偶尔打印 m 的值，观察训练过程中 ABCAS 的动态调整效果
            print(f"ABCAS m: {m:.4f}")
        # if torch.is_autocast_enabled():
        #     try:
        #         # pytorch >2.4
        #         target_dtype = torch.get_autocast_dtype('cuda')
        #     except AttributeError:
        #         # 兼容旧版本的 get_autocast_gpu_dtype
        #         target_dtype = torch.get_autocast_gpu_dtype()
        # else:
        #     # 未开启自动混合精度（纯 FP32 状态）
        #     target_dtype = x.dtype
        # scaled_weight = (self.conv.weight * m).to(device=x.device, dtype=target_dtype)
        # bias = self.conv.bias.to(device=x.device, dtype=target_dtype) if self.conv.bias is not None else None
        # return F.conv2d(
        #     x, scaled_weight, bias,
        #     stride=self.conv.stride,
        #     padding=self.conv.padding,
        #     dilation=self.conv.dilation,
        #     groups=self.conv.groups
        # )
        out = self.conv(x)
        
        # 对输出乘以 m，在数学上完全等价于对权重 W 和偏置 b 同时乘上 m
        return out * m

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

    def __init__(self, image_channels: int = 3, feature_maps: int = 64, use_spectral_norm: bool = False, scale_sn: bool = False) -> None:
        super().__init__()
        ndf = feature_maps
        self.current_scale = 1.0
        scale_fn = lambda: self.current_scale
        bias_enabled = use_spectral_norm
        def _maybe_abcas_sn(module: nn.Conv2d, sn: bool, scale: bool) -> nn.Module:
            if sn and scale:
                return ScaledSpectralNormConv2d(module, scale_fn)
            elif sn:
                return spectral_norm(module)
            else:
                return module

        self.net = nn.Sequential(
            # 输入: N x 3 x 64 x 64
            _maybe_abcas_sn(
                nn.Conv2d(image_channels, ndf, kernel_size=4, stride=2, padding=1, bias=bias_enabled),
                use_spectral_norm,
                scale_sn,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            # N x ndf x 32 x 32
            _maybe_abcas_sn(
                nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=bias_enabled),
                use_spectral_norm,
                scale_sn,
            ),
            norm_act(ndf * 2, use_spectral_norm),
            # nn.BatchNorm2d(ndf * 2),
            # nn.LeakyReLU(0.2, inplace=True),
            # N x (ndf*2) x 16 x 16
            _maybe_abcas_sn(
                nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=bias_enabled),
                use_spectral_norm,
                scale_sn,
            ),
            norm_act(ndf * 4, use_spectral_norm),
            # nn.BatchNorm2d(ndf * 4),
            # nn.LeakyReLU(0.2, inplace=True),
            # N x (ndf*4) x 8 x 8
            _maybe_abcas_sn(
                nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=bias_enabled),
                use_spectral_norm,
                scale_sn,
            ),
            norm_act(ndf * 8, use_spectral_norm),
            # nn.BatchNorm2d(ndf * 8),
            # nn.LeakyReLU(0.2, inplace=True),
            # N x (ndf*8) x 4 x 4
            _maybe_abcas_sn(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=bias_enabled),
                use_spectral_norm,
                scale_sn,
            ),
            # 输出真假 logits，训练时配合 BCEWithLogitsLoss，兼容 mixed precision。
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """输入图片张量，输出每张图片对应的真假 logits。"""

        return self.net(image).view(-1)




class PixelNorm(nn.Module):
    """对潜变量做逐样本归一化，缓解 StyleGAN 映射网络输入尺度不稳定的问题。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对每个样本的通道维度做归一化，保持潜变量尺度稳定。"""

        return x * torch.rsqrt(torch.mean(x.pow(2), dim=1, keepdim=True) + 1e-8)



def init_dcgan_weights(module: nn.Module) -> None:
    """DCGAN 论文推荐的初始化方式。"""

    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        weight = module.weight_orig if hasattr(module, "weight_orig") else module.weight
        nn.init.normal_(weight.data, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias.data)
            
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)