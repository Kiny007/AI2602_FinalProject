import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gan_faces.utils import get_device, load_generator_from_checkpoint, make_noise, save_generated_grid, set_random_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用训练好的 GAN 生成头像图片")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-images", type=int, default=64)
    parser.add_argument("--output", type=str, default="outputs/generated/grid.png")
    parser.add_argument("--nrow", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)
    device = get_device(args.device)

    generator, model_args, _ = load_generator_from_checkpoint(args.checkpoint, device)
    latent_dim = int(model_args.get("latent_dim", 100))

    # 一次性生成指定数量的头像，并保存为图片网格。
    with torch.no_grad():
        noise = make_noise(args.num_images, latent_dim, device)
        images = generator(noise)
    save_generated_grid(images, args.output, nrow=args.nrow)
    print(f"生成图片已保存到: {args.output}")


if __name__ == "__main__":
    main()
