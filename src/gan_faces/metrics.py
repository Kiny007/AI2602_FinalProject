from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torchvision.models import Inception_V3_Weights, inception_v3

from .utils import make_noise


def _normalize_for_inception(images: torch.Tensor) -> torch.Tensor:
    """把 [0, 1] 图片转换为 Inception v3 预训练模型需要的 ImageNet 归一化。"""

    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


def _score_probabilities(all_probs: torch.Tensor, splits: int) -> tuple[float, float]:
    """根据 Inception 分类概率计算 IS 的均值和标准差。"""

    split_scores = []
    for split_probs in torch.chunk(all_probs, splits):
        py = split_probs.mean(dim=0, keepdim=True)
        kl = split_probs * (split_probs.log() - py.log())
        split_scores.append(torch.exp(kl.sum(dim=1).mean()).item())

    mean = float(torch.tensor(split_scores).mean().item())
    std = float(torch.tensor(split_scores).std(unbiased=False).item())
    return mean, std


@torch.no_grad()
def inception_score_from_images(
    image_batches: Iterable[torch.Tensor],
    num_images: int,
    splits: int,
    device: torch.device,
) -> tuple[float, float]:
    """对已经生成的图片批次计算 Inception Score。

    每个批次的图片应为 RGB 张量，像素范围为 [-1, 1]。
    """

    weights = Inception_V3_Weights.DEFAULT
    inception = inception_v3(weights=weights, transform_input=False)
    inception.to(device)
    inception.eval()

    probs = []
    seen = 0

    for images in image_batches:
        if seen >= num_images:
            break

        images = images[: num_images - seen].to(device)
        seen += images.size(0)

        # 生成器输出为 [-1, 1]，评估前还原到 [0, 1] 并缩放到 299x299。
        images = (images + 1.0).mul(0.5).clamp(0.0, 1.0)
        images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
        images = _normalize_for_inception(images)

        logits = inception(images)
        if hasattr(logits, "logits"):
            logits = logits.logits
        probs.append(torch.softmax(logits, dim=1).cpu())

    if seen < num_images:
        raise ValueError(f"用于 IS 评估的图片数量不足: 需要 {num_images}, 实际 {seen}")

    all_probs = torch.cat(probs, dim=0)
    return _score_probabilities(all_probs, splits)


@torch.no_grad()
def inception_score(
    generator: torch.nn.Module,
    latent_dim: int,
    num_images: int,
    batch_size: int,
    splits: int,
    device: torch.device,
) -> tuple[float, float]:
    """计算 Inception Score。

    IS 越高通常表示生成图片越清晰且类别分布越丰富。对人脸这种单类别数据，
    IS 的解释能力有限，但它满足项目“FID 或 IS”中的基础评估要求。
    """

    total_batches = math.ceil(num_images / batch_size)

    def generated_batches() -> Iterable[torch.Tensor]:
        for batch_index in range(total_batches):
            current_batch = min(batch_size, num_images - batch_index * batch_size)
            z = make_noise(current_batch, latent_dim, device)
            yield generator(z)

    return inception_score_from_images(generated_batches(), num_images, splits, device)
