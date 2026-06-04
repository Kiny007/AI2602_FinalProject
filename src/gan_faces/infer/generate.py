"""生成图片入口。

本脚本加载训练好的 DCGAN 或 StyleGAN-Lite checkpoint，采样随机潜变量，
并把生成的人头图像保存为网格图，常用于训练后快速检查模型效果。
"""

import argparse
import sys
from pathlib import Path

import torch


SRC_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_ROOT))

from gan_faces.utils import get_device, load_generator_from_checkpoint, make_noise, save_generated_grid, set_random_seed


def parse_args() -> argparse.Namespace:
    """解析生成图片所需的 checkpoint、图片数量和输出路径参数。"""

    parser = argparse.ArgumentParser(description="使用训练好的 GAN 生成头像图片")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-images", type=int, default=64)
    parser.add_argument("--output", type=str, default="outputs/generated/grid.png")
    parser.add_argument("--nrow", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    """加载生成器 checkpoint，采样噪声并保存生成图片网格。"""

    args = parse_args()
    set_random_seed(args.seed)
    device = get_device(args.device)

    generator, model_args, _ = load_generator_from_checkpoint(args.checkpoint, device)
    # checkpoint 中保存了 latent_dim；旧 checkpoint 缺失时默认使用 100。
    latent_dim = int(model_args.get("latent_dim", 100))

    # 一次性生成指定数量的头像，并保存为图片网格。
    with torch.no_grad():
        noise = make_noise(args.num_images, latent_dim, device)
        images = generator(noise)
    save_generated_grid(images, args.output, nrow=args.nrow)
    print(f"生成图片已保存到: {args.output}")


if __name__ == "__main__":
    main()
