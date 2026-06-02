"""DCGAN 与 StyleGAN-Lite 性能对比入口。

本脚本加载两个已经训练好的生成器 checkpoint，统一评估生成器参数量、
生成速度和 Inception Score，并把结果保存为 JSON/CSV 供实验报告引用。
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gan_faces.metrics import inception_score
from gan_faces.utils import (
    count_parameters,
    ensure_dir,
    get_device,
    load_generator_from_checkpoint,
    make_noise,
    save_json,
    set_random_seed,
)


def parse_args() -> argparse.Namespace:
    """解析 DCGAN 和 StyleGAN-Lite 对比实验的命令行参数。"""

    parser = argparse.ArgumentParser(description="对比 DCGAN 与轻量 StyleGAN 风格模型的性能差异")
    parser.add_argument("--dcgan-checkpoint", type=str, required=True)
    parser.add_argument("--stylegan-checkpoint", type=str, required=True)
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-json", type=str, default="outputs/metrics/model_comparison.json")
    parser.add_argument("--output-csv", type=str, default="outputs/metrics/model_comparison.csv")
    return parser.parse_args()


@torch.no_grad()
def measure_generation_speed(
    generator: torch.nn.Module,
    latent_dim: int,
    num_images: int,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    """测量生成速度，返回总耗时和每秒生成图片数。"""

    generator.eval()
    generated = 0

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    while generated < num_images:
        current_batch = min(batch_size, num_images - generated)
        noise = make_noise(current_batch, latent_dim, device)
        _ = generator(noise)
        generated += current_batch

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    images_per_second = num_images / max(elapsed, 1e-8)
    return elapsed, images_per_second


def write_csv(rows: list[dict[str, float | int | str]], output_path: str | Path) -> None:
    """保存 CSV，方便在实验报告中整理表格。"""

    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    fieldnames = [
        "name",
        "model_type",
        "checkpoint",
        "parameters",
        "generation_seconds",
        "images_per_second",
        "is_mean",
        "is_std",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_one(
    name: str,
    checkpoint_path: str,
    num_images: int,
    batch_size: int,
    splits: int,
    device: torch.device,
) -> dict[str, float | int | str]:
    """对单个模型计算复杂度、速度和 Inception Score。"""

    generator, model_args, checkpoint = load_generator_from_checkpoint(checkpoint_path, device)
    latent_dim = int(model_args.get("latent_dim", 100))
    model_type = checkpoint.get("model_type", "dcgan")

    seconds, images_per_second = measure_generation_speed(
        generator=generator,
        latent_dim=latent_dim,
        num_images=num_images,
        batch_size=batch_size,
        device=device,
    )
    is_mean, is_std = inception_score(
        generator=generator,
        latent_dim=latent_dim,
        num_images=num_images,
        batch_size=batch_size,
        splits=splits,
        device=device,
    )

    return {
        "name": name,
        "model_type": model_type,
        "checkpoint": checkpoint_path,
        "parameters": count_parameters(generator),
        "generation_seconds": seconds,
        "images_per_second": images_per_second,
        "is_mean": is_mean,
        "is_std": is_std,
    }


def main() -> None:
    """依次评估两个模型，并把汇总结果写入文件。"""

    args = parse_args()
    set_random_seed(args.seed)
    device = get_device(args.device)

    rows = [
        # 两个模型使用相同 num_images、batch_size 和 splits，保证指标可比。
        evaluate_one(
            name="DCGAN",
            checkpoint_path=args.dcgan_checkpoint,
            num_images=args.num_images,
            batch_size=args.batch_size,
            splits=args.splits,
            device=device,
        ),
        evaluate_one(
            name="StyleGAN-Lite",
            checkpoint_path=args.stylegan_checkpoint,
            num_images=args.num_images,
            batch_size=args.batch_size,
            splits=args.splits,
            device=device,
        ),
    ]

    result = {
        "metric": "DCGAN vs StyleGAN-Lite comparison",
        "num_images": args.num_images,
        "batch_size": args.batch_size,
        "splits": args.splits,
        "device": str(device),
        "rows": rows,
    }
    save_json(result, args.output_json)
    write_csv(rows, args.output_csv)

    print("模型对比结果:")
    print("name, model_type, parameters, images/s, IS mean, IS std")
    for row in rows:
        print(
            f"{row['name']}, {row['model_type']}, {row['parameters']}, "
            f"{row['images_per_second']:.2f}, {row['is_mean']:.4f}, {row['is_std']:.4f}"
        )
    print(f"JSON 已保存到: {args.output_json}")
    print(f"CSV 已保存到: {args.output_csv}")


if __name__ == "__main__":
    main()
