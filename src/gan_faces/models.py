import torch
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


def init_dcgan_weights(module: nn.Module) -> None:
    """DCGAN 论文推荐的初始化方式。"""

    classname = module.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)
