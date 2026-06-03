"""课程项目的 NVIDIA 指标评测入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

SRC_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_ROOT))

from gan_faces.eval.nvidia_evaluator import evaluate_adapter, parse_metrics, supported_metrics
from gan_faces.utils import get_device, load_generator_from_checkpoint, save_json, set_random_seed


class AI2602GeneratorAdapter(nn.Module):
    """把课程项目生成器适配到 NVIDIA 指标接口。"""

    def __init__(self, generator: nn.Module, z_dim: int, img_resolution: int, img_channels: int) -> None:
        super().__init__()
        self.generator = generator
        self.z_dim = z_dim
        self.c_dim = 0
        self.img_resolution = img_resolution
        self.img_channels = img_channels

    @classmethod
    def from_generator(cls, generator: nn.Module, z_dim: int, device: torch.device) -> "AI2602GeneratorAdapter":
        with torch.no_grad():
            sample_z = torch.randn(1, z_dim, device=device)
            sample_img = generator(sample_z)
        if sample_img.ndim != 4:
            raise ValueError(f"生成器输出维度异常: {tuple(sample_img.shape)}")
        return cls(
            generator=generator,
            z_dim=z_dim,
            img_resolution=int(sample_img.shape[-1]),
            img_channels=int(sample_img.shape[1]),
        )

    def forward(self, z: torch.Tensor, c: torch.Tensor | None = None, **_kwargs) -> torch.Tensor:
        del c
        if z.ndim > 2:
            z = z.view(z.size(0), -1)
        return self.generator(z)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 NVIDIA StyleGAN2-ADA 标准指标评估生成器")
    parser.add_argument("--checkpoint", type=str, required=True, help="课程项目训练得到的 checkpoint 路径")
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="真实图像目录或 zip 路径，建议传入 CelebA 图片目录",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="fid5k",
        help=(
            "逗号分隔的 NVIDIA 指标名，例如 fid5k,kid5k,pr5k3,is5k；"
            "兼容旧别名 fid/is/both"
        ),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--verbose", action="store_true", help="打印 NVIDIA 指标计算进度")
    parser.add_argument("--no-cache", action="store_true", help="禁用真实特征缓存")
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def build_adapter_from_checkpoint(checkpoint_path: str | Path, device: torch.device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint_path}")

    generator, model_args, checkpoint = load_generator_from_checkpoint(checkpoint_path, device)
    latent_dim = int(model_args.get("latent_dim", 100))
    adapter = AI2602GeneratorAdapter.from_generator(generator, z_dim=latent_dim, device=device)
    adapter.eval().requires_grad_(False).to(device)
    return adapter, checkpoint


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)
    device = get_device(args.device)
    metrics = parse_metrics(args.metrics)

    adapter, checkpoint = build_adapter_from_checkpoint(args.checkpoint, device)
    result = evaluate_adapter(
        adapter=adapter,
        data_path=args.data_root,
        metrics=metrics,
        verbose=args.verbose,
        cache=not args.no_cache,
    )
    result["checkpoint"] = str(args.checkpoint)
    result["model_type"] = checkpoint.get("model_type", "dcgan")
    result["metric_backend"] = "nvidia_stylegan2_ada"
    result["requested_metrics"] = metrics
    result["device"] = str(device)
    
    if args.output_json is None:
        output_json = Path(args.checkpoint).parent.parent / "metrics" / "nvidia_metrics.json"
    else:
        output_json = args.output_json
    save_json(result, output_json)
    print("可用指标: " + ", ".join(supported_metrics()))
    print(f"评测完成，结果已保存到: {output_json}")
    for metric_name, metric_values in result["metrics"].items():
        print(f"{metric_name}: {metric_values}")


if __name__ == "__main__":
    main()

# python evaluate.py --checkpoint outputs/dcgan_bce_sigmoid/checkpoints/latest.pt --data-root ".\data\celeba\img_align_celeba" --metrics fid5k --device cuda --verbose
