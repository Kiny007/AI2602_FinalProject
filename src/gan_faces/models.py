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
        return self.net(image).view(-1)


class PixelNorm(nn.Module):
    """对潜变量做逐样本归一化，缓解 StyleGAN 映射网络输入尺度不稳定的问题。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        if z.ndim == 4:
            z = z.flatten(1)
        return self.net(z)


class NoiseInjection(nn.Module):
    """给特征图加入逐像素噪声，用于模拟 StyleGAN 中的随机细节变化。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
