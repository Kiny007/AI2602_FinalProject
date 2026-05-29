import argparse
import csv
import sys
from pathlib import Path

import torch
from torch import nn, optim


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gan_faces.data import build_dataloader, build_unpaired_dataset
from gan_faces.models import CycleGenerator, PatchDiscriminator, init_cyclegan_weights
from gan_faces.utils import ensure_dir, get_device, save_generated_grid, set_random_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 CycleGAN 无配对图像域转换模型")
    parser.add_argument("--domain-a-root", type=str, required=True, help="源域 A 图片目录")
    parser.add_argument("--domain-b-root", type=str, required=True, help="目标域 B 图片目录")
    parser.add_argument("--output-dir", type=str, default="outputs/cyclegan")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--generator-features", type=int, default=64)
    parser.add_argument("--discriminator-features", type=int, default=64)
    parser.add_argument("--num-residual-blocks", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--lambda-cycle", type=float, default=10.0)
    parser.add_argument("--lambda-identity", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def write_log_header(log_path: Path) -> None:
    if not log_path.exists():
        with log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "step",
                    "loss_g",
                    "loss_d_a",
                    "loss_d_b",
                    "loss_cycle",
                    "loss_identity",
                    "loss_gan_a2b",
                    "loss_gan_b2a",
                ]
            )


def append_log(log_path: Path, row: list[float | int]) -> None:
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def set_requires_grad(models: list[nn.Module], requires_grad: bool) -> None:
    for model in models:
        for parameter in model.parameters():
            parameter.requires_grad = requires_grad


def discriminator_loss(
    discriminator: nn.Module,
    criterion: nn.Module,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
) -> torch.Tensor:
    real_logits = discriminator(real_images)
    fake_logits = discriminator(fake_images.detach())
    loss_real = criterion(real_logits, torch.ones_like(real_logits))
    loss_fake = criterion(fake_logits, torch.zeros_like(fake_logits))
    return 0.5 * (loss_real + loss_fake)


@torch.no_grad()
def save_cycle_samples(
    generator_a2b: nn.Module,
    generator_b2a: nn.Module,
    fixed_a: torch.Tensor,
    fixed_b: torch.Tensor,
    output_path: Path,
) -> None:
    generator_a2b.eval()
    generator_b2a.eval()

    fake_b = generator_a2b(fixed_a)
    rec_a = generator_b2a(fake_b)
    fake_a = generator_b2a(fixed_b)
    rec_b = generator_a2b(fake_a)
    grid = torch.cat([fixed_a, fake_b, rec_a, fixed_b, fake_a, rec_b], dim=0)
    save_generated_grid(grid, output_path, nrow=fixed_a.size(0))


def main() -> None:
    args = parse_args()
    if args.image_size != 64:
        raise ValueError("当前 CycleGAN 结构按 64x64 图片配置，请保持 --image-size 64")

    set_random_seed(args.seed)
    device = get_device(args.device)
    torch.backends.cudnn.benchmark = device.type == "cuda"

    output_dir = ensure_dir(args.output_dir)
    sample_dir = ensure_dir(output_dir / "samples")
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    log_path = output_dir / "train_log.csv"
    write_log_header(log_path)

    dataset = build_unpaired_dataset(args.domain_a_root, args.domain_b_root, args.image_size)
    drop_last = len(dataset) >= args.batch_size
    dataloader = build_dataloader(dataset, args.batch_size, args.workers, drop_last=drop_last)
    print(
        f"CycleGAN 数据域: A={args.domain_a_root}, B={args.domain_b_root}, "
        f"样本步数: {len(dataset)}, 设备: {device}"
    )

    fixed_a, fixed_b = next(iter(dataloader))
    sample_count = min(4, fixed_a.size(0), fixed_b.size(0))
    fixed_a = fixed_a[:sample_count].to(device)
    fixed_b = fixed_b[:sample_count].to(device)

    generator_args = {
        "image_channels": 3,
        "feature_maps": args.generator_features,
        "num_residual_blocks": args.num_residual_blocks,
    }
    disc_args = {
        "image_channels": 3,
        "feature_maps": args.discriminator_features,
    }

    generator_a2b = CycleGenerator(**generator_args).to(device)
    generator_b2a = CycleGenerator(**generator_args).to(device)
    discriminator_a = PatchDiscriminator(**disc_args).to(device)
    discriminator_b = PatchDiscriminator(**disc_args).to(device)

    generator_a2b.apply(init_cyclegan_weights)
    generator_b2a.apply(init_cyclegan_weights)
    discriminator_a.apply(init_cyclegan_weights)
    discriminator_b.apply(init_cyclegan_weights)

    criterion_gan = nn.MSELoss()
    criterion_cycle = nn.L1Loss()

    optimizer_g = optim.Adam(
        list(generator_a2b.parameters()) + list(generator_b2a.parameters()),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
    )
    optimizer_d = optim.Adam(
        list(discriminator_a.parameters()) + list(discriminator_b.parameters()),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
    )

    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        if checkpoint.get("model_type") != "cyclegan":
            raise ValueError("resume checkpoint 不是 CycleGAN 模型")
        generator_a2b.load_state_dict(checkpoint["generator_a2b"])
        generator_b2a.load_state_dict(checkpoint["generator_b2a"])
        discriminator_a.load_state_dict(checkpoint["discriminator_a"])
        discriminator_b.load_state_dict(checkpoint["discriminator_b"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"已从 {args.resume} 恢复训练，将从 epoch {start_epoch} 开始")

    for epoch in range(start_epoch, args.epochs + 1):
        generator_a2b.train()
        generator_b2a.train()
        discriminator_a.train()
        discriminator_b.train()

        for step, (real_a, real_b) in enumerate(dataloader, start=1):
            real_a = real_a.to(device, non_blocking=True)
            real_b = real_b.to(device, non_blocking=True)

            set_requires_grad([discriminator_a, discriminator_b], False)
            optimizer_g.zero_grad(set_to_none=True)

            fake_b = generator_a2b(real_a)
            rec_a = generator_b2a(fake_b)
            fake_a = generator_b2a(real_b)
            rec_b = generator_a2b(fake_a)

            logits_fake_b = discriminator_b(fake_b)
            logits_fake_a = discriminator_a(fake_a)
            loss_gan_a2b = criterion_gan(logits_fake_b, torch.ones_like(logits_fake_b))
            loss_gan_b2a = criterion_gan(logits_fake_a, torch.ones_like(logits_fake_a))
            loss_cycle = criterion_cycle(rec_a, real_a) + criterion_cycle(rec_b, real_b)

            if args.lambda_identity > 0:
                same_a = generator_b2a(real_a)
                same_b = generator_a2b(real_b)
                loss_identity = criterion_cycle(same_a, real_a) + criterion_cycle(same_b, real_b)
            else:
                loss_identity = torch.zeros((), device=device)

            loss_g = (
                loss_gan_a2b
                + loss_gan_b2a
                + args.lambda_cycle * loss_cycle
                + args.lambda_identity * loss_identity
            )
            loss_g.backward()
            optimizer_g.step()

            set_requires_grad([discriminator_a, discriminator_b], True)
            optimizer_d.zero_grad(set_to_none=True)
            loss_d_a = discriminator_loss(discriminator_a, criterion_gan, real_a, fake_a)
            loss_d_b = discriminator_loss(discriminator_b, criterion_gan, real_b, fake_b)
            loss_d = loss_d_a + loss_d_b
            loss_d.backward()
            optimizer_d.step()

            if step == 1 or step % 50 == 0:
                print(
                    f"Epoch [{epoch}/{args.epochs}] Step [{step}/{len(dataloader)}] "
                    f"Loss_G={loss_g.item():.4f} Loss_D_A={loss_d_a.item():.4f} "
                    f"Loss_D_B={loss_d_b.item():.4f} Cycle={loss_cycle.item():.4f}"
                )
                append_log(
                    log_path,
                    [
                        epoch,
                        step,
                        loss_g.item(),
                        loss_d_a.item(),
                        loss_d_b.item(),
                        loss_cycle.item(),
                        loss_identity.item(),
                        loss_gan_a2b.item(),
                        loss_gan_b2a.item(),
                    ],
                )

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            save_cycle_samples(
                generator_a2b,
                generator_b2a,
                fixed_a,
                fixed_b,
                sample_dir / f"epoch_{epoch:04d}.png",
            )

        state = {
            "model_type": "cyclegan",
            "epoch": epoch,
            "generator_a2b": generator_a2b.state_dict(),
            "generator_b2a": generator_b2a.state_dict(),
            "discriminator_a": discriminator_a.state_dict(),
            "discriminator_b": discriminator_b.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "generator_args": generator_args,
            "disc_args": disc_args,
            "train_args": vars(args),
        }
        torch.save(state, checkpoint_dir / "latest.pt")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(state, checkpoint_dir / f"cyclegan_epoch_{epoch:04d}.pt")

    print("CycleGAN 训练完成。")


if __name__ == "__main__":
    main()
