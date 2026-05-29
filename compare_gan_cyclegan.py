"""DCGAN 与 CycleGAN 性能对比入口。

DCGAN 是从随机噪声生成图片的无条件生成模型；CycleGAN 是从源域图片
翻译到目标域的图像到图像模型。本脚本把二者的参数量、速度和 IS 放到
同一张表中，同时额外计算 CycleGAN 的循环重建误差 cycle_l1。
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gan_faces.data import build_dataloader, build_unpaired_dataset
from gan_faces.metrics import inception_score, inception_score_from_images
from gan_faces.utils import (
    count_parameters,
    ensure_dir,
    get_device,
    load_cyclegan_generators_from_checkpoint,
    load_generator_from_checkpoint,
    make_noise,
    save_json,
    set_random_seed,
)


def parse_args() -> argparse.Namespace:
    """解析基础 GAN 与 CycleGAN 对比实验的命令行参数。"""

    parser = argparse.ArgumentParser(description="对比基础 DCGAN 与 CycleGAN 的性能差异")
    parser.add_argument("--dcgan-checkpoint", type=str, required=True)
    parser.add_argument("--cyclegan-checkpoint", type=str, required=True)
    parser.add_argument("--domain-a-root", type=str, required=True, help="CycleGAN A 域图片目录")
    parser.add_argument("--domain-b-root", type=str, required=True, help="CycleGAN B 域图片目录")
    parser.add_argument("--direction", choices=["a2b", "b2a"], default="a2b")
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-json", type=str, default="outputs/metrics/gan_vs_cyclegan.json")
    parser.add_argument("--output-csv", type=str, default="outputs/metrics/gan_vs_cyclegan.csv")
    return parser.parse_args()


def count_state_dict_parameters(state_dict: dict[str, torch.Tensor]) -> int:
    """统计 checkpoint 中某个 state_dict 的参数总量。"""

    return sum(tensor.numel() for tensor in state_dict.values())


@torch.no_grad()
def measure_dcgan_speed(
    generator: torch.nn.Module,
    latent_dim: int,
    num_images: int,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    """测量 DCGAN 从随机噪声生成图片的推理速度。"""

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
    return elapsed, num_images / max(elapsed, 1e-8)


def collect_domain_batches(
    dataloader: torch.utils.data.DataLoader,
    domain: str,
    num_images: int,
) -> list[torch.Tensor]:
    """预先取出评估图片，避免把磁盘读取时间计入生成速度。"""

    batches: list[torch.Tensor] = []
    seen = 0
    domain_index = 0 if domain == "a" else 1

    while seen < num_images:
        for batch in dataloader:
            images = batch[domain_index]
            remaining = num_images - seen
            images = images[:remaining].clone()
            batches.append(images)
            seen += images.size(0)
            if seen >= num_images:
                break

    return batches


@torch.no_grad()
def measure_image_to_image_speed(
    generator: torch.nn.Module,
    source_batches: list[torch.Tensor],
    device: torch.device,
) -> tuple[float, float]:
    """测量 CycleGAN 对已有图片做图像到图像翻译的推理速度。"""

    generator.eval()
    num_images = sum(batch.size(0) for batch in source_batches)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    for batch in source_batches:
        images = batch.to(device, non_blocking=True)
        _ = generator(images)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed, num_images / max(elapsed, 1e-8)


@torch.no_grad()
def measure_cycle_l1(
    forward_generator: torch.nn.Module,
    backward_generator: torch.nn.Module,
    source_batches: list[torch.Tensor],
    device: torch.device,
) -> float:
    """计算 CycleGAN 的平均循环重建 L1 误差，数值越低表示重建越接近源图。"""

    forward_generator.eval()
    backward_generator.eval()
    total = 0.0
    count = 0

    for batch in source_batches:
        source = batch.to(device, non_blocking=True)
        reconstructed = backward_generator(forward_generator(source))
        per_image = torch.mean(torch.abs(reconstructed - source), dim=(1, 2, 3))
        total += per_image.sum().item()
        count += source.size(0)

    return total / max(count, 1)


def translated_batches(
    generator: torch.nn.Module,
    source_batches: list[torch.Tensor],
    device: torch.device,
):
    """按批次生成 CycleGAN 翻译图片，供 IS 评估函数流式消费。"""

    with torch.no_grad():
        for batch in source_batches:
            yield generator(batch.to(device, non_blocking=True))


def write_csv(rows: list[dict[str, float | int | str]], output_path: str | Path) -> None:
    """把对比结果保存为 CSV 表格，便于复制到实验报告。"""

    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    fieldnames = [
        "name",
        "model_type",
        "checkpoint",
        "input_type",
        "direction",
        "parameters",
        "total_parameters",
        "generation_seconds",
        "images_per_second",
        "is_mean",
        "is_std",
        "cycle_l1",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """执行 DCGAN 和指定方向 CycleGAN 的完整对比评估。"""

    args = parse_args()
    set_random_seed(args.seed)
    device = get_device(args.device)

    dataset = build_unpaired_dataset(args.domain_a_root, args.domain_b_root, args.image_size)
    dataloader = build_dataloader(dataset, args.batch_size, args.workers, drop_last=False)
    source_domain = "a" if args.direction == "a2b" else "b"
    # 预取源域图片，使速度测试主要反映模型推理而不是磁盘读取。
    source_batches = collect_domain_batches(dataloader, source_domain, args.num_images)

    # DCGAN 分支：随机噪声 -> 生成图片。
    dcgan, dcgan_args, dcgan_checkpoint = load_generator_from_checkpoint(args.dcgan_checkpoint, device)
    latent_dim = int(dcgan_args.get("latent_dim", 100))
    dcgan_seconds, dcgan_ips = measure_dcgan_speed(
        generator=dcgan,
        latent_dim=latent_dim,
        num_images=args.num_images,
        batch_size=args.batch_size,
        device=device,
    )
    dcgan_is_mean, dcgan_is_std = inception_score(
        generator=dcgan,
        latent_dim=latent_dim,
        num_images=args.num_images,
        batch_size=args.batch_size,
        splits=args.splits,
        device=device,
    )
    dcgan_total_parameters = count_parameters(dcgan)
    if "discriminator" in dcgan_checkpoint:
        dcgan_total_parameters += count_state_dict_parameters(dcgan_checkpoint["discriminator"])

    # CycleGAN 分支：真实源域图片 -> 目标域翻译图片。
    generator_a2b, generator_b2a, _, cycle_checkpoint = load_cyclegan_generators_from_checkpoint(
        args.cyclegan_checkpoint,
        device,
    )
    if args.direction == "a2b":
        # A->B 时，forward 负责翻译，backward 负责循环重建回 A。
        cycle_forward = generator_a2b
        cycle_backward = generator_b2a
        cycle_name = "CycleGAN A->B"
    else:
        # B->A 时，forward/backward 的方向相反。
        cycle_forward = generator_b2a
        cycle_backward = generator_a2b
        cycle_name = "CycleGAN B->A"

    cycle_seconds, cycle_ips = measure_image_to_image_speed(cycle_forward, source_batches, device)
    cycle_is_mean, cycle_is_std = inception_score_from_images(
        translated_batches(cycle_forward, source_batches, device),
        num_images=args.num_images,
        splits=args.splits,
        device=device,
    )
    cycle_l1 = measure_cycle_l1(cycle_forward, cycle_backward, source_batches, device)
    cycle_total_parameters = sum(
        count_state_dict_parameters(cycle_checkpoint[key])
        for key in ["generator_a2b", "generator_b2a", "discriminator_a", "discriminator_b"]
    )

    rows: list[dict[str, float | int | str]] = [
        {
            "name": "DCGAN",
            "model_type": dcgan_checkpoint.get("model_type", "dcgan"),
            "checkpoint": args.dcgan_checkpoint,
            "input_type": "random_noise",
            "direction": "-",
            "parameters": count_parameters(dcgan),
            "total_parameters": dcgan_total_parameters,
            "generation_seconds": dcgan_seconds,
            "images_per_second": dcgan_ips,
            "is_mean": dcgan_is_mean,
            "is_std": dcgan_is_std,
            "cycle_l1": "",
        },
        {
            "name": cycle_name,
            "model_type": "cyclegan",
            "checkpoint": args.cyclegan_checkpoint,
            "input_type": f"image_domain_{source_domain}",
            "direction": args.direction,
            "parameters": count_parameters(cycle_forward),
            "total_parameters": cycle_total_parameters,
            "generation_seconds": cycle_seconds,
            "images_per_second": cycle_ips,
            "is_mean": cycle_is_mean,
            "is_std": cycle_is_std,
            "cycle_l1": cycle_l1,
        },
    ]

    result = {
        # notes 说明两个模型任务定义不同，避免报告中误读指标含义。
        "metric": "DCGAN vs CycleGAN comparison",
        "num_images": args.num_images,
        "batch_size": args.batch_size,
        "splits": args.splits,
        "direction": args.direction,
        "device": str(device),
        "notes": "DCGAN 从随机噪声生成图片；CycleGAN 对真实源域图片做图像到图像翻译。",
        "rows": rows,
    }
    save_json(result, args.output_json)
    write_csv(rows, args.output_csv)

    print("基础 GAN 与 CycleGAN 对比结果:")
    print("name, parameters, total_parameters, images/s, IS mean, IS std, cycle_l1")
    for row in rows:
        cycle_l1_text = row["cycle_l1"] if row["cycle_l1"] != "" else "-"
        print(
            f"{row['name']}, {row['parameters']}, {row['total_parameters']}, "
            f"{row['images_per_second']:.2f}, {row['is_mean']:.4f}, "
            f"{row['is_std']:.4f}, {cycle_l1_text}"
        )
    print(f"JSON 已保存到: {args.output_json}")
    print(f"CSV 已保存到: {args.output_csv}")


if __name__ == "__main__":
    main()
