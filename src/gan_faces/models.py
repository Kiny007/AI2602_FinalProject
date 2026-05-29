"""GAN 模型结构定义。

本文件集中定义项目中用到的神经网络模块：
DCGAN 生成器/判别器、CycleGAN 生成器/PatchGAN 判别器，以及轻量
StyleGAN 风格生成器。训练脚本只负责组织训练流程，模型细节统一放在这里。
"""

import torch
import torch.nn.functional as F
from torch import nn


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

    def __init__(self, image_channels: int = 3, feature_maps: int = 64) -> None:
        super().__init__()
        ndf = feature_maps

        self.net = nn.Sequential(
            # 输入: N x 3 x 64 x 64
            nn.Conv2d(image_channels, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # N x ndf x 32 x 32
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # N x (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # N x (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # N x (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # 输出 logits，不加 Sigmoid，配合 BCEWithLogitsLoss 更稳定。
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """输入图片张量，输出每张图片对应的真假 logits。"""

        return self.net(image).view(-1)


class CycleResidualBlock(nn.Module):
    """CycleGAN 生成器中的残差块，保持特征图尺寸不变。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回残差连接后的特征图，用于增强 CycleGAN 生成器表达能力。"""

        return x + self.block(x)


class CycleGenerator(nn.Module):
    """CycleGAN 的 ResNet 生成器，用于无配对图像域转换。

    输入和输出都是 64x64 RGB 图像，像素范围为 [-1, 1]。与 DCGAN 不同，
    CycleGAN 生成器不是从噪声采样，而是把源域图片翻译到目标域。
    """

    def __init__(
        self,
        image_channels: int = 3,
        feature_maps: int = 64,
        num_residual_blocks: int = 6,
    ) -> None:
        super().__init__()
        ngf = feature_maps

        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(image_channels, ngf, kernel_size=7, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(ngf, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf * 2, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf * 4, affine=True),
            nn.ReLU(inplace=True),
        ]

        layers.extend(CycleResidualBlock(ngf * 4) for _ in range(num_residual_blocks))
        layers.extend(
            [
                nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ngf * 2, affine=True),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ngf, affine=True),
                nn.ReLU(inplace=True),
                nn.ReflectionPad2d(3),
                nn.Conv2d(ngf, image_channels, kernel_size=7, stride=1, padding=0),
                nn.Tanh(),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """把源域图片翻译为目标域图片，输出范围仍为 [-1, 1]。"""

        return self.net(image)


class PatchDiscriminator(nn.Module):
    """CycleGAN 使用的 PatchGAN 判别器，输出局部 patch 的真假 logits。"""

    def __init__(self, image_channels: int = 3, feature_maps: int = 64) -> None:
        super().__init__()
        ndf = feature_maps

        self.net = nn.Sequential(
            nn.Conv2d(image_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """输出 patch 级别的真假 logits，而不是单个全图真假分数。"""

        return self.net(image)


class PixelNorm(nn.Module):
    """对潜变量做逐样本归一化，缓解 StyleGAN 映射网络输入尺度不稳定的问题。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对每个样本的通道维度做归一化，保持潜变量尺度稳定。"""

        return x * torch.rsqrt(torch.mean(x.pow(2), dim=1, keepdim=True) + 1e-8)


class MappingNetwork(nn.Module):
    """StyleGAN 的映射网络：把 z 空间映射到更适合控制风格的 w 空间。"""

    def __init__(self, latent_dim: int = 100, style_dim: int = 128, layers: int = 4) -> None:
        super().__init__()
        modules: list[nn.Module] = [PixelNorm()]
        in_features = latent_dim

        for _ in range(layers):
            modules.extend(
                [
                    nn.Linear(in_features, style_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            in_features = style_dim

        self.net = nn.Sequential(*modules)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """把原始随机噪声 z 映射到 StyleGAN 使用的风格向量 w。"""

        if z.ndim == 4:
            z = z.flatten(1)
        return self.net(z)


class NoiseInjection(nn.Module):
    """给特征图加入逐像素噪声，用于模拟 StyleGAN 中的随机细节变化。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """给每个样本、每个空间位置注入可学习强度的随机噪声。"""

        noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        return x + self.weight * noise


class AdaptiveInstanceNorm(nn.Module):
    """AdaIN：用风格向量控制每个通道的缩放和平移。"""

    def __init__(self, channels: int, style_dim: int) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False, eps=1e-8)
        self.style = nn.Linear(style_dim, channels * 2)
        nn.init.normal_(self.style.weight, 0.0, 0.02)
        nn.init.zeros_(self.style.bias)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """根据 style 向量生成通道缩放和平移参数，并调制输入特征。"""

        style_params = self.style(style).view(style.size(0), 2, x.size(1), 1, 1)
        gamma = style_params[:, 0] + 1.0
        beta = style_params[:, 1]
        return gamma * self.norm(x) + beta


class StyledConvBlock(nn.Module):
    """StyleGAN 风格卷积块：上采样、卷积、噪声注入、AdaIN。"""

    def __init__(self, in_channels: int, out_channels: int, style_dim: int, upsample: bool) -> None:
        super().__init__()
        self.upsample = upsample
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.noise = NoiseInjection(out_channels)
        self.adain = AdaptiveInstanceNorm(out_channels, style_dim)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """执行可选上采样、卷积、噪声注入和 AdaIN 风格调制。"""

        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        x = self.noise(x)
        x = self.adain(x, style)
        return self.activation(x)


class StyleGeneratorLite(nn.Module):
    """轻量 StyleGAN 风格生成器，用于 Bonus 中与 DCGAN 做性能对比。

    该实现保留 StyleGAN 的关键思想：映射网络、学习常量输入、AdaIN 风格调制和噪声注入。
    为了适合课程项目和 64x64 CelebA 实验，这里没有实现完整 StyleGAN/StyleGAN2 的所有技巧。
    """

    def __init__(
        self,
        latent_dim: int = 100,
        image_channels: int = 3,
        feature_maps: int = 64,
        style_dim: int = 128,
        mapping_layers: int = 4,
    ) -> None:
        super().__init__()
        ngf = feature_maps
        channels = [ngf * 8, ngf * 8, ngf * 4, ngf * 2, ngf]

        self.mapping = MappingNetwork(latent_dim, style_dim, mapping_layers)
        self.constant = nn.Parameter(torch.randn(1, channels[0], 4, 4))
        self.blocks = nn.ModuleList(
            [
                StyledConvBlock(channels[0], channels[0], style_dim, upsample=False),
                StyledConvBlock(channels[0], channels[1], style_dim, upsample=True),
                StyledConvBlock(channels[1], channels[2], style_dim, upsample=True),
                StyledConvBlock(channels[2], channels[3], style_dim, upsample=True),
                StyledConvBlock(channels[3], channels[4], style_dim, upsample=True),
            ]
        )
        self.to_rgb = nn.Sequential(
            nn.Conv2d(channels[4], image_channels, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """从噪声生成头像图片，内部先映射风格向量再逐层调制特征。"""

        style = self.mapping(z)
        x = self.constant.repeat(z.size(0), 1, 1, 1)
        for block in self.blocks:
            x = block(x, style)
        return self.to_rgb(x)


def init_dcgan_weights(module: nn.Module) -> None:
    """DCGAN 论文推荐的初始化方式。"""

    classname = module.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)


def init_stylegan_lite_weights(module: nn.Module) -> None:
    """轻量 StyleGAN 生成器的卷积/线性层初始化。"""

    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias.data)


def init_cyclegan_weights(module: nn.Module) -> None:
    """CycleGAN 论文常用的卷积/归一化层初始化。"""

    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias.data)
    elif isinstance(module, nn.InstanceNorm2d) and module.affine:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.zeros_(module.bias.data)
