import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torchvision.utils import save_image

from .models import Generator, StyleGeneratorLite


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在，并返回 Path 对象。"""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_random_seed(seed: int) -> None:
    """固定随机种子，让同一配置下的实验更容易复现。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> torch.device:
    """根据参数选择设备，auto 会优先使用 CUDA。"""

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def make_noise(batch_size: int, latent_dim: int, device: torch.device) -> torch.Tensor:
    """生成标准正态噪声，作为生成器输入。"""

    return torch.randn(batch_size, latent_dim, 1, 1, device=device)


def save_generated_grid(images: torch.Tensor, output_path: str | Path, nrow: int = 8) -> None:
    """保存图片网格；输入图片范围为 [-1, 1]。"""

    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    save_image(images, output_path, nrow=nrow, normalize=True, value_range=(-1, 1))


def save_json(data: dict[str, Any], output_path: str | Path) -> None:
    """保存 JSON 结果，便于后续写实验报告。"""

    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def build_generator(model_type: str, model_args: dict[str, Any]) -> nn.Module:
    """根据 checkpoint 中的模型类型创建对应生成器。"""

    if model_type == "dcgan":
        return Generator(**model_args)
    if model_type == "stylegan_lite":
        return StyleGeneratorLite(**model_args)
    raise ValueError(f"未知生成器类型: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """统计可训练参数量，用于模型复杂度对比。"""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def load_generator_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    """从训练 checkpoint 中恢复生成器。"""

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = checkpoint.get("model_args", {})
    # 旧版 DCGAN checkpoint 没有 model_type 字段，因此默认按 dcgan 处理。
    model_type = checkpoint.get("model_type", "dcgan")
    generator = build_generator(model_type, model_args)
    generator.load_state_dict(checkpoint["generator"])
    generator.to(device)
    generator.eval()
    return generator, model_args, checkpoint
