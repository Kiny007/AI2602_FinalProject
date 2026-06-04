"""`stylegan2-pytorch` 模型的 NVIDIA 指标评测入口。"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

SRC_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STYLEGAN2_ROOT = PROJECT_ROOT.parent / "stylegan2-pytorch"
sys.path.insert(0, str(SRC_ROOT))

from gan_faces.eval.nvidia_evaluator import evaluate_adapter, parse_metrics, supported_metrics
from gan_faces.utils import get_device, save_json, set_random_seed


def _add_stylegan2_repo_to_path(stylegan2_root: Path) -> None:
    if str(stylegan2_root) not in sys.path:
        sys.path.insert(0, str(stylegan2_root))


def _infer_checkpoint_spec(checkpoint_path: Path) -> dict[str, Any]:
    match = re.fullmatch(r"model_(\d+)", checkpoint_path.stem)
    if match is None:
        raise ValueError(f"无法从 checkpoint 文件名推断编号: {checkpoint_path.name}")
    if checkpoint_path.parent.parent == checkpoint_path.parent:
        raise ValueError(f"checkpoint 路径结构异常: {checkpoint_path}")

    model_name = checkpoint_path.parent.name
    models_dir_path = checkpoint_path.parent.parent
    base_dir = models_dir_path.parent
    return {
        "base_dir": base_dir,
        "models_dir": models_dir_path.name,
        "name": model_name,
        "load_from": int(match.group(1)),
    }


class StyleGAN2PytorchAdapter(nn.Module):
    """把 lucidrains 的 `stylegan2-pytorch` 生成器适配到 NVIDIA 指标接口。"""

    def __init__(
        self,
        style_vectorizer: nn.Module,
        generator: nn.Module,
        truncate_style_fn,
        styles_def_to_tensor_fn,
        image_noise_fn,
        z_dim: int,
        img_resolution: int,
        img_channels: int,
        num_layers: int,
        trunc_psi: float,
        noise_device,
    ) -> None:
        super().__init__()
        self.style_vectorizer = style_vectorizer
        self.generator = generator
        self._truncate_style = truncate_style_fn
        self._styles_def_to_tensor = styles_def_to_tensor_fn
        self._image_noise = image_noise_fn
        self.z_dim = z_dim
        self.c_dim = 0
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.num_layers = num_layers
        self.trunc_psi = trunc_psi
        self.noise_device = noise_device

    def forward(self, z: torch.Tensor, c: torch.Tensor | None = None, **_kwargs) -> torch.Tensor:
        del c
        if z.ndim > 2:
            z = z.view(z.size(0), -1)

        styles = self.style_vectorizer(z)
        if self.trunc_psi >= 0:
            styles = self._truncate_style(styles, S=self.style_vectorizer, trunc_psi=self.trunc_psi)

        w_styles = self._styles_def_to_tensor([(styles, self.num_layers)])
        noise = self._image_noise(z.size(0), self.img_resolution, device=self.noise_device)
        images = self.generator(w_styles, noise).clamp_(0.0, 1.0)
        return images.mul(2.0).sub(1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 NVIDIA StyleGAN2-ADA 指标评估 stylegan2-pytorch 生成器")
    parser.add_argument("--checkpoint", type=str, required=True, help="`stylegan2-pytorch` 的 `model_*.pt` 路径")
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="真实图像目录或 zip 路径，建议传入 CelebA 图片目录",
    )
    parser.add_argument(
        "--stylegan2-root",
        type=str,
        default=str(DEFAULT_STYLEGAN2_ROOT),
        help="`stylegan2-pytorch` 仓库根目录",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="fid5k",
        help="逗号分隔的指标名，例如 fid5k,kid5k,pr5k3,is5k；兼容 fid/is/both",
    )
    parser.add_argument("--trunc-psi", type=float, default=0.75, help="使用 EMA 生成器时的 truncation psi")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--verbose", action="store_true", help="打印 NVIDIA 指标计算进度")
    parser.add_argument("--no-cache", action="store_true", help="禁用真实特征缓存")
    parser.add_argument(
        "--output-json",
        type=str,
        default="outputs/metrics/stylegan2_pytorch_nvidia_metrics.json",
        help="评测结果输出路径",
    )
    return parser.parse_args()


def build_stylegan2_pytorch_adapter(
    checkpoint_path: str | Path,
    stylegan2_root: str | Path,
    device: torch.device,
    trunc_psi: float,
):
    if device.type != "cuda":
        raise ValueError("`stylegan2-pytorch` 评测当前仅支持 CUDA 设备")

    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint_path}")

    stylegan2_root = Path(stylegan2_root).resolve()
    if not stylegan2_root.exists():
        raise FileNotFoundError(f"`stylegan2-pytorch` 仓库不存在: {stylegan2_root}")

    _add_stylegan2_repo_to_path(stylegan2_root)
    from stylegan2_pytorch.stylegan2_pytorch import Trainer, image_noise, styles_def_to_tensor

    checkpoint_spec = _infer_checkpoint_spec(checkpoint_path)
    trainer = Trainer(
        name=checkpoint_spec["name"],
        base_dir=str(checkpoint_spec["base_dir"]),
        models_dir=checkpoint_spec["models_dir"],
    )
    trainer.load_config()
    trainer.steps = checkpoint_spec["load_from"] * trainer.save_every
    load_data = torch.load(checkpoint_path, map_location=device, weights_only=True)
    trainer.GAN.load_state_dict(load_data["GAN"])
    trainer.GAN.eval()
    trainer.GAN.requires_grad_(False)
    trainer.trunc_psi = trunc_psi

    style_vectorizer = trainer.GAN.SE.to(device).eval()
    generator = trainer.GAN.GE.to(device).eval()
    adapter = StyleGAN2PytorchAdapter(
        style_vectorizer=style_vectorizer,
        generator=generator,
        truncate_style_fn=trainer.truncate_style,
        styles_def_to_tensor_fn=styles_def_to_tensor,
        image_noise_fn=image_noise,
        z_dim=int(generator.latent_dim),
        img_resolution=int(generator.image_size),
        img_channels=4 if bool(trainer.transparent) else 3,
        num_layers=int(generator.num_layers),
        trunc_psi=trunc_psi,
        noise_device=device,
    )
    adapter.eval().requires_grad_(False).to(device)
    return adapter, checkpoint_spec


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)
    device = get_device(args.device)
    metrics = parse_metrics(args.metrics)

    adapter, checkpoint_spec = build_stylegan2_pytorch_adapter(
        checkpoint_path=args.checkpoint,
        stylegan2_root=args.stylegan2_root,
        device=device,
        trunc_psi=args.trunc_psi,
    )

    result = evaluate_adapter(
        adapter=adapter,
        data_path=args.data_root,
        metrics=metrics,
        verbose=args.verbose,
        cache=not args.no_cache,
    )
    result["checkpoint"] = str(Path(args.checkpoint).resolve())
    result["stylegan2_root"] = str(Path(args.stylegan2_root).resolve())
    result["checkpoint_spec"] = {
        "base_dir": str(checkpoint_spec["base_dir"]),
        "models_dir": checkpoint_spec["models_dir"],
        "name": checkpoint_spec["name"],
        "load_from": checkpoint_spec["load_from"],
    }
    result["model_type"] = "stylegan2-pytorch"
    result["metric_backend"] = "nvidia_stylegan2_ada"
    result["requested_metrics"] = metrics
    result["device"] = str(device)
    result["trunc_psi"] = args.trunc_psi

    save_json(result, args.output_json)
    print("可用指标: " + ", ".join(supported_metrics()))
    print(f"评测完成，结果已保存到: {args.output_json}")
    for metric_name, metric_values in result["metrics"].items():
        print(f"{metric_name}: {metric_values}")


if __name__ == "__main__":
    main()

