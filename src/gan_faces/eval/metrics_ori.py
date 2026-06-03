"""生成质量评估指标。

本文件实现 Inception Score 和 FID，用于评估生成图像的清晰度、多样性
以及生成分布与真实图像分布的距离。
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import Inception_V3_Weights, inception_v3

from .utils import make_noise


def _normalize_for_inception(images: torch.Tensor) -> torch.Tensor:
    """把 [0, 1] 图片转换为 Inception v3 预训练模型需要的 ImageNet 归一化。"""

    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


def _prepare_images_for_inception(images: torch.Tensor) -> torch.Tensor:
    """把 [-1, 1] RGB 图片转换为 Inception v3 可直接输入的张量。"""

    images = (images + 1.0).mul(0.5).clamp(0.0, 1.0)
    images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
    return _normalize_for_inception(images)


def _build_inception_classifier(device: torch.device) -> nn.Module:
    """构建用于 IS 的 Inception v3 分类器。"""

    weights = Inception_V3_Weights.DEFAULT
    inception = inception_v3(weights=weights, transform_input=False)
    inception.to(device)
    inception.eval()
    return inception


def _build_inception_feature_extractor(device: torch.device) -> nn.Module:
    """构建用于 FID 的 Inception v3 特征提取器，输出 2048 维 pool 特征。"""

    weights = Inception_V3_Weights.DEFAULT
    inception = inception_v3(weights=weights, transform_input=False)
    inception.fc = nn.Identity()
    inception.to(device)
    inception.eval()
    return inception


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


def _feature_statistics(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """计算特征均值和协方差矩阵。"""

    if features.size(0) < 2:
        raise ValueError("FID 至少需要 2 张图片来估计协方差")

    features = features.double()
    mean = features.mean(dim=0)
    centered = features - mean
    covariance = centered.t().mm(centered) / (features.size(0) - 1)
    return mean, covariance


def _symmetric_matrix_sqrt(matrix: torch.Tensor) -> torch.Tensor:
    """对对称半正定矩阵计算平方根矩阵。"""

    matrix = (matrix + matrix.t()) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    eigenvalues = torch.clamp(eigenvalues, min=0.0)
    return (eigenvectors * torch.sqrt(eigenvalues).unsqueeze(0)).mm(eigenvectors.t())


def _frechet_distance(real_features: torch.Tensor, generated_features: torch.Tensor) -> float:
    """根据两组 Inception 特征计算 Frechet Inception Distance。"""

    real_mean, real_cov = _feature_statistics(real_features)
    generated_mean, generated_cov = _feature_statistics(generated_features)

    mean_diff = real_mean - generated_mean
    sqrt_real_cov = _symmetric_matrix_sqrt(real_cov)
    covmean = _symmetric_matrix_sqrt(sqrt_real_cov.mm(generated_cov).mm(sqrt_real_cov))

    fid = (
        mean_diff.dot(mean_diff)
        + torch.trace(real_cov)
        + torch.trace(generated_cov)
        - 2.0 * torch.trace(covmean)
    )
    return float(torch.clamp(fid, min=0.0).item())


@torch.no_grad()
def _collect_inception_features(
    image_batches: Iterable[torch.Tensor],
    num_images: int,
    device: torch.device,
    inception: nn.Module,
) -> torch.Tensor:
    """流式提取指定数量图片的 Inception pool 特征。"""

    features = []
    seen = 0

    for images in image_batches:
        if seen >= num_images:
            break

        images = images[: num_images - seen].to(device, non_blocking=True)
        seen += images.size(0)
        images = _prepare_images_for_inception(images)

        batch_features = inception(images)
        if hasattr(batch_features, "logits"):
            batch_features = batch_features.logits
        features.append(batch_features.flatten(1).cpu())

    if seen < num_images:
        raise ValueError(f"用于 FID 评估的图片数量不足: 需要 {num_images}, 实际 {seen}")

    return torch.cat(features, dim=0)


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

    inception = _build_inception_classifier(device)
    probs = []
    seen = 0

    for images in image_batches:
        if seen >= num_images:
            break

        images = images[: num_images - seen].to(device)
        seen += images.size(0)

        # 生成器输出为 [-1, 1]，评估前还原到 [0, 1] 并缩放到 299x299。
        images = _prepare_images_for_inception(images)
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


@torch.no_grad()
def frechet_inception_distance_from_images(
    real_image_batches: Iterable[torch.Tensor],
    generated_image_batches: Iterable[torch.Tensor],
    num_images: int,
    device: torch.device,
) -> float:
    """对真实图片批次和生成图片批次计算 FID，数值越低通常越好。

    两个输入流中的图片都应为 RGB 张量，像素范围为 [-1, 1]。
    """

    inception = _build_inception_feature_extractor(device)
    real_features = _collect_inception_features(real_image_batches, num_images, device, inception)
    generated_features = _collect_inception_features(
        generated_image_batches,
        num_images,
        device,
        inception,
    )
    return _frechet_distance(real_features, generated_features)


@torch.no_grad()
def frechet_inception_distance(
    generator: torch.nn.Module,
    latent_dim: int,
    real_image_batches: Iterable[torch.Tensor],
    num_images: int,
    batch_size: int,
    device: torch.device,
) -> float:
    """从生成器采样图片并与真实图片计算 FID。"""

    generator.eval()
    total_batches = math.ceil(num_images / batch_size)

    def generated_batches() -> Iterable[torch.Tensor]:
        for batch_index in range(total_batches):
            current_batch = min(batch_size, num_images - batch_index * batch_size)
            z = make_noise(current_batch, latent_dim, device)
            yield generator(z)

    return frechet_inception_distance_from_images(
        real_image_batches=real_image_batches,
        generated_image_batches=generated_batches(),
        num_images=num_images,
        device=device,
    )
