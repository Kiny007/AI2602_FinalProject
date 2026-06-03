"""DCGAN 训练入口。

本脚本负责读取人脸数据集、构建 DCGAN 生成器和判别器、执行对抗训练，
并周期性保存训练日志、样例图片和 checkpoint。
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Sequence, Union

import torch
from torch import nn, optim


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gan_faces.data import build_dataloader, build_dataset
from gan_faces.models import Discriminator, Generator, init_dcgan_weights
from gan_faces.tensorboard import (
    add_sample_images,
    add_training_scalars,
    close_summary_writer,
    create_summary_writer,
)
from gan_faces.utils import ensure_dir, get_device, make_noise, save_generated_grid, set_random_seed


def parse_args() -> argparse.Namespace:
    """解析 DCGAN 训练所需的命令行参数。"""

    parser = argparse.ArgumentParser(description="训练 DCGAN 人头图像生成模型")
    parser.add_argument("--dataset", choices=["folder", "lfw", "celeba"], default="folder")
    parser.add_argument("--data-root", type=str, default="data/faces")
    parser.add_argument("--download", action="store_true", help="允许 torchvision 下载数据集")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--generator-features", type=int, default=64)
    parser.add_argument("--discriminator-features", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--tensorboard-dir", type=str, default="", help="TensorBoard 日志目录，默认 output-dir/tensorboard")
    parser.add_argument("--no-tensorboard", action="store_true", help="关闭 TensorBoard 日志写入")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def write_log_header(log_path: Path) -> None:
    """创建训练日志 CSV，并写入表头。"""

    if not log_path.exists():
        with log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "step", "loss_d", "loss_g", "d_real", "d_fake"])


def append_log(log_path: Path, row: Sequence[Union[float, int]]) -> None:
    """向训练日志追加一行损失和判别器输出统计。"""

    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def main() -> None:
    """组织完整 DCGAN 训练流程：数据、模型、优化器、训练循环和保存。"""

    args = parse_args()
    if args.image_size != 64:
        raise ValueError("当前 DCGAN 结构固定输出 64x64 图片，请保持 --image-size 64")
    print(args.data_root)
    set_random_seed(args.seed)
    device = get_device(args.device)
    torch.backends.cudnn.benchmark = device.type == "cuda"

    # 所有实验产物统一写入 output_dir，便于后续生成图片和评估脚本复用。
    output_dir = ensure_dir(args.output_dir)
    sample_dir = ensure_dir(output_dir / "samples")
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    log_path = output_dir / "train_log.csv"
    write_log_header(log_path)
    tensorboard_dir = ensure_dir(args.tensorboard_dir or output_dir / "tensorboard")
    writer = None
    if not args.no_tensorboard:
        writer = create_summary_writer(tensorboard_dir)
        print(f"TensorBoard 日志目录: {tensorboard_dir}")
        print(f"查看命令: tensorboard --logdir {tensorboard_dir}")

    dataset = build_dataset(args.dataset, args.data_root, args.image_size, download=args.download)
    drop_last = len(dataset) >= args.batch_size
    dataloader = build_dataloader(dataset, args.batch_size, args.workers, drop_last=drop_last)
    print(f"数据集: {args.dataset}, 图片数量: {len(dataset)}, 设备: {device}")

    model_args = {
        "latent_dim": args.latent_dim,
        "image_channels": 3,
        "feature_maps": args.generator_features,
    }
    disc_args = {
        "image_channels": 3,
        "feature_maps": args.discriminator_features,
    }

    generator = Generator(**model_args).to(device)
    discriminator = Discriminator(**disc_args).to(device)
    generator.apply(init_dcgan_weights)
    discriminator.apply(init_dcgan_weights)

    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))


    start_epoch = 1
    if args.resume:
        # resume 时同时恢复模型和优化器，保证动量状态不丢失。
        checkpoint = torch.load(args.resume, map_location=device)
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"已从 {args.resume} 恢复训练，将从 epoch {start_epoch} 开始")

    # 固定一组噪声，方便比较不同 epoch 生成效果的变化。
    fixed_noise = make_noise(min(64, args.batch_size), args.latent_dim, device)

    for epoch in range(start_epoch, args.epochs + 1):
        generator.train()
        discriminator.train()

        for step, real_images in enumerate(dataloader, start=1):
            real_images = real_images.to(device, non_blocking=True)
            batch_size = real_images.size(0)

            # 训练判别器：真实图片应判为 1，生成图片应判为 0。
            real_targets = torch.ones(batch_size, device=device)
            fake_targets = torch.zeros(batch_size, device=device)

            optimizer_d.zero_grad(set_to_none=True)
            real_scores = discriminator(real_images)
            loss_d_real = criterion(real_scores, real_targets)

            noise = make_noise(batch_size, args.latent_dim, device)
            fake_images = generator(noise)
            fake_scores = discriminator(fake_images.detach())
            loss_d_fake = criterion(fake_scores, fake_targets)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optimizer_d.step()

            # 训练生成器：希望判别器把生成图片也判为真实。
            optimizer_g.zero_grad(set_to_none=True)
            fool_targets = torch.ones(batch_size, device=device)
            fake_scores_for_g = discriminator(fake_images)
            loss_g = criterion(fake_scores_for_g, fool_targets)
            loss_g.backward()
            optimizer_g.step()

            global_step = (epoch - 1) * len(dataloader) + step
            loss_d_value = loss_d.item()
            loss_g_value = loss_g.item()
            d_real = real_scores.mean().item()
            d_fake = fake_scores.mean().item()
            add_training_scalars(
                writer=writer,
                global_step=global_step,
                loss_d=loss_d_value,
                loss_g=loss_g_value,
                d_real=d_real,
                d_fake=d_fake,
                lr_g=optimizer_g.param_groups[0]["lr"],
                lr_d=optimizer_d.param_groups[0]["lr"],
            )

            if step == 1 or step % 50 == 0:
                print(
                    f"Epoch [{epoch}/{args.epochs}] Step [{step}/{len(dataloader)}] "
                    f"Loss_D={loss_d_value:.4f} Loss_G={loss_g_value:.4f} "
                    f"D(real)={d_real:.4f} D(fake)={d_fake:.4f}"
                )
                append_log(
                    log_path,
                    [epoch, step, loss_d_value, loss_g_value, d_real, d_fake],
                )

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            # 使用固定噪声保存样例图，可以直观看到训练过程中的质量变化。
            generator.eval()
            with torch.no_grad():
                samples = generator(fixed_noise)
            save_generated_grid(samples, sample_dir / f"epoch_{epoch:04d}.png", nrow=8)
            add_sample_images(writer, samples, epoch, nrow=8)

        state = {
            # checkpoint 保存模型结构参数和训练参数，方便跨脚本加载。
            "model_type": "dcgan",
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "model_args": model_args,
            "disc_args": disc_args,
            "train_args": vars(args),
        }
        torch.save(state, checkpoint_dir / "latest.pt")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(state, checkpoint_dir / f"dcgan_epoch_{epoch:04d}.pt")

    close_summary_writer(writer)
    print("训练完成。")


if __name__ == "__main__":
    main()
