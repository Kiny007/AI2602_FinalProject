"""StyleGAN-Lite 训练入口。

本脚本沿用 DCGAN 的判别器和训练框架，但把生成器替换为轻量
StyleGAN 风格结构，用于与基础 DCGAN 做模型复杂度和生成质量对比。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch import nn, optim


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gan_faces.data import build_dataloader, build_dataset
from gan_faces.models import Discriminator, StyleGeneratorLite, init_dcgan_weights, init_stylegan_lite_weights
from gan_faces.utils import ensure_dir, get_device, make_noise, save_generated_grid, set_random_seed


def parse_args() -> argparse.Namespace:
    """解析 StyleGAN-Lite 训练相关的命令行参数。"""

    parser = argparse.ArgumentParser(description="训练轻量 StyleGAN 风格人头图像生成模型")
    parser.add_argument("--dataset", choices=["folder", "lfw", "celeba"], default="folder")
    parser.add_argument("--data-root", type=str, default="data/faces")
    parser.add_argument("--download", action="store_true", help="允许 torchvision 下载数据集")
    parser.add_argument("--output-dir", type=str, default="outputs/stylegan_lite")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--generator-features", type=int, default=64)
    parser.add_argument("--discriminator-features", type=int, default=64)
    parser.add_argument("--style-dim", type=int, default=128)
    parser.add_argument("--mapping-layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.0)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def write_log_header(log_path: Path) -> None:
    """创建训练日志 CSV，并写入字段名。"""

    if not log_path.exists():
        with log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "step", "loss_d", "loss_g", "d_real", "d_fake"])


def append_log(log_path: Path, row: list[float | int]) -> None:
    """向训练日志追加当前 step 的损失和判别器输出。"""

    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def main() -> None:
    """组织 StyleGAN-Lite 的数据加载、模型初始化、训练和保存流程。"""

    args = parse_args()
    if args.image_size != 64:
        raise ValueError("当前轻量 StyleGAN 结构固定输出 64x64 图片，请保持 --image-size 64")

    set_random_seed(args.seed)
    device = get_device(args.device)
    torch.backends.cudnn.benchmark = device.type == "cuda"

    # StyleGAN-Lite 单独放在 outputs/stylegan_lite，避免和 DCGAN checkpoint 混淆。
    output_dir = ensure_dir(args.output_dir)
    sample_dir = ensure_dir(output_dir / "samples")
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    log_path = output_dir / "train_log.csv"
    write_log_header(log_path)

    dataset = build_dataset(args.dataset, args.data_root, args.image_size, download=args.download)
    drop_last = len(dataset) >= args.batch_size
    dataloader = build_dataloader(dataset, args.batch_size, args.workers, drop_last=drop_last)
    print(f"数据集: {args.dataset}, 图片数量: {len(dataset)}, 设备: {device}")

    model_args = {
        "latent_dim": args.latent_dim,
        "image_channels": 3,
        "feature_maps": args.generator_features,
        "style_dim": args.style_dim,
        "mapping_layers": args.mapping_layers,
    }
    disc_args = {
        "image_channels": 3,
        "feature_maps": args.discriminator_features,
    }

    generator = StyleGeneratorLite(**model_args).to(device)
    discriminator = Discriminator(**disc_args).to(device)
    generator.apply(init_stylegan_lite_weights)
    discriminator.apply(init_dcgan_weights)

    criterion = nn.BCEWithLogitsLoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    start_epoch = 1
    if args.resume:
        # 只允许从 StyleGAN-Lite checkpoint 恢复，防止误加载 DCGAN 权重。
        checkpoint = torch.load(args.resume, map_location=device)
        if checkpoint.get("model_type") != "stylegan_lite":
            raise ValueError("resume checkpoint 不是 stylegan_lite 模型")
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"已从 {args.resume} 恢复训练，将从 epoch {start_epoch} 开始")

    # 固定噪声用于观察同一组潜变量在训练过程中的生成质量变化。
    fixed_noise = make_noise(min(64, args.batch_size), args.latent_dim, device)

    for epoch in range(start_epoch, args.epochs + 1):
        generator.train()
        discriminator.train()

        for step, real_images in enumerate(dataloader, start=1):
            real_images = real_images.to(device, non_blocking=True)
            batch_size = real_images.size(0)

            # 训练判别器：真实图像标签做轻微平滑，降低判别器过强带来的震荡。
            real_targets = torch.full((batch_size,), 0.9, device=device)
            fake_targets = torch.zeros(batch_size, device=device)

            optimizer_d.zero_grad(set_to_none=True)
            real_logits = discriminator(real_images)
            loss_d_real = criterion(real_logits, real_targets)

            noise = make_noise(batch_size, args.latent_dim, device)
            fake_images = generator(noise)
            fake_logits = discriminator(fake_images.detach())
            loss_d_fake = criterion(fake_logits, fake_targets)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optimizer_d.step()

            # 训练生成器：让生成图像尽量骗过同一个判别器。
            optimizer_g.zero_grad(set_to_none=True)
            fool_targets = torch.ones(batch_size, device=device)
            fake_logits_for_g = discriminator(fake_images)
            loss_g = criterion(fake_logits_for_g, fool_targets)
            loss_g.backward()
            optimizer_g.step()

            if step == 1 or step % 50 == 0:
                d_real = torch.sigmoid(real_logits).mean().item()
                d_fake = torch.sigmoid(fake_logits).mean().item()
                print(
                    f"Epoch [{epoch}/{args.epochs}] Step [{step}/{len(dataloader)}] "
                    f"Loss_D={loss_d.item():.4f} Loss_G={loss_g.item():.4f} "
                    f"D(real)={d_real:.4f} D(fake)={d_fake:.4f}"
                )
                append_log(
                    log_path,
                    [epoch, step, loss_d.item(), loss_g.item(), d_real, d_fake],
                )

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            # 固定潜变量样例用于横向观察 StyleGAN-Lite 的训练进展。
            generator.eval()
            with torch.no_grad():
                samples = generator(fixed_noise)
            save_generated_grid(samples, sample_dir / f"epoch_{epoch:04d}.png", nrow=8)

        state = {
            # 记录 model_type，后续 generate/evaluate 脚本会据此重建正确生成器。
            "model_type": "stylegan_lite",
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
            torch.save(state, checkpoint_dir / f"stylegan_lite_epoch_{epoch:04d}.pt")

    print("轻量 StyleGAN 风格模型训练完成。")


if __name__ == "__main__":
    main()
