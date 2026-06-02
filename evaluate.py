"""Inception Score 评估入口。

本脚本加载训练好的生成器，批量生成图片，并调用 metrics.py 中的
Inception Score 实现评估生成图像质量，结果同时打印并保存为 JSON。
"""

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gan_faces.data import build_dataloader, build_dataset
from gan_faces.metrics import frechet_inception_distance, inception_score
from gan_faces.utils import get_device, load_generator_from_checkpoint, save_json, set_random_seed


def parse_args() -> argparse.Namespace:
    """解析 IS 评估所需的 checkpoint、采样数量和输出路径参数。"""

    parser = argparse.ArgumentParser(description="使用 IS 或 FID 评估 GAN 生成头像质量")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--metric", choices=["is", "fid", "both"], default="is")
    parser.add_argument("--num-images", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--dataset", choices=["folder", "lfw", "celeba"], default="folder")
    parser.add_argument("--data-root", type=str, default="data/faces")
    parser.add_argument("--download", action="store_true", help="允许 torchvision 下载真实数据集")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-json", type=str, default="outputs/metrics/is_score.json")
    return parser.parse_args()


def repeat_real_batches(dataloader: torch.utils.data.DataLoader):
    """循环提供真实图片批次，支持真实数据集小于评估图片数的情况。"""

    while True:
        yielded = False
        for images in dataloader:
            yielded = True
            yield images
        if not yielded:
            return


def main() -> None:
    """执行生成器加载、指标计算和评估结果保存。"""

    args = parse_args()
    set_random_seed(args.seed)
    device = get_device(args.device)

    generator, model_args, checkpoint = load_generator_from_checkpoint(args.checkpoint, device)
    # 不同生成器可以有不同潜变量维度，因此从 checkpoint 的 model_args 中读取。
    latent_dim = int(model_args.get("latent_dim", 100))

    result = {
        "num_images": args.num_images,
        "model_type": checkpoint.get("model_type", "dcgan"),
        "checkpoint": args.checkpoint,
    }

    if args.metric in {"is", "both"}:
        mean, std = inception_score(
            generator=generator,
            latent_dim=latent_dim,
            num_images=args.num_images,
            batch_size=args.batch_size,
            splits=args.splits,
            device=device,
        )
        if args.metric == "is":
            result.update(
                {
                    "metric": "Inception Score",
                    "mean": mean,
                    "std": std,
                    "splits": args.splits,
                }
            )
        else:
            result["inception_score"] = {
                "mean": mean,
                "std": std,
                "splits": args.splits,
            }
        print(f"Inception Score: mean={mean:.4f}, std={std:.4f}")

    if args.metric in {"fid", "both"}:
        dataset = build_dataset(args.dataset, args.data_root, args.image_size, download=args.download)
        dataloader = build_dataloader(
            dataset=dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            shuffle=False,
            drop_last=False,
        )
        if len(dataset) < args.num_images:
            print(
                f"真实数据集只有 {len(dataset)} 张图片，FID 将循环复用真实图片直到 {args.num_images} 张。"
            )
        fid = frechet_inception_distance(
            generator=generator,
            latent_dim=latent_dim,
            real_image_batches=repeat_real_batches(dataloader),
            num_images=args.num_images,
            batch_size=args.batch_size,
            device=device,
        )
        result["fid"] = fid
        result["real_dataset"] = {
            "dataset": args.dataset,
            "data_root": args.data_root,
            "image_size": args.image_size,
        }
        if args.metric == "fid":
            result["metric"] = "FID"
        else:
            result["metric"] = "Inception Score + FID"
        print(f"FID: {fid:.4f}")

    save_json(result, args.output_json)
    print(f"评估结果已保存到: {args.output_json}")


if __name__ == "__main__":
    main()
