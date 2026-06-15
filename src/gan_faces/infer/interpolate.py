"""潜变量插值入口。

本脚本加载训练好的生成器，使用两个固定随机种子生成潜变量端点，
再在潜变量空间中做线性或球面插值，观察生成头像是否平滑变化。
"""

import argparse
import sys
from pathlib import Path

import torch


SRC_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_ROOT))

from gan_faces.latent_interpolation import linear_interpolate, spherical_interpolate
from gan_faces.utils import get_device, load_generator_from_checkpoint, save_generated_grid


def parse_args() -> argparse.Namespace:
    """解析插值端点 seed、插值步数和输出路径参数。"""

    parser = argparse.ArgumentParser(description="在两张生成头像之间做潜变量插值")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed-a", type=int, default=7)
    parser.add_argument("--seed-b", type=int, default=99)
    parser.add_argument(
        "--method",
        type=str,
        choices=["linear", "spherical"],
        default="linear",
        help="linear 为线性插值；spherical 为球面插值，适合 Gaussian z 空间",
    )
    parser.add_argument("--output", type=str, default="outputs/interpolation/linear.png")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def latent_from_seed(seed: int, latent_dim: int) -> torch.Tensor:
    """用固定 seed 生成端点潜变量，便于复现实验图。"""

    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, latent_dim, generator=generator)


def main() -> None:
    """执行潜变量插值，并保存一行连续变化的头像网格。"""

    args = parse_args()
    if args.steps < 2:
        raise ValueError("--steps 至少为 2，才能包含两个端点")

    device = get_device(args.device)
    generator, model_args, _ = load_generator_from_checkpoint(args.checkpoint, device)
    latent_dim = int(model_args.get("latent_dim", 100))

    z_a = latent_from_seed(args.seed_a, latent_dim).to(device)
    z_b = latent_from_seed(args.seed_b, latent_dim).to(device)

    alphas = torch.linspace(0.0, 1.0, steps=args.steps, device=device).view(args.steps, 1)
    if args.method == "linear":
        z = linear_interpolate(z_a, z_b, alphas)
    else:
        z = spherical_interpolate(z_a, z_b, alphas)

    with torch.no_grad():
        images = generator(z)
    save_generated_grid(images, args.output, nrow=args.steps)
    print(f"插值结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
