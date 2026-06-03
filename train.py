"""DCGAN 训练入口。

在原有课程项目逻辑上补充了更工程化的训练运行时：多卡 DDP、子进程管理、
EMA 生成器、JSONL/TensorBoard 日志，以及统一的样例图和 checkpoint 快照。
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data.distributed import DistributedSampler


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gan_faces.data import build_dataloader, build_dataset
from gan_faces.models import Discriminator, Generator, init_dcgan_weights
from gan_faces.config_utils import apply_config_defaults, load_yaml_config
from gan_faces.eval.evaluate import AI2602GeneratorAdapter
from gan_faces.eval.nvidia_evaluator import evaluate_adapter, parse_metrics
from gan_faces.train_runtime import (
    TrainingLogger,
    average_tensor,
    barrier,
    cleanup_training_process,
    cleanup_distributed,
    create_training_layout,
    find_free_port,
    install_signal_handlers,
    interrupt_requested,
    maybe_wrap_ddp,
    prepare_device,
    restore_signal_handlers,
    save_training_options,
    seed_everything,
    setup_distributed,
    unwrap_module,
    update_ema,
)
from gan_faces.utils import count_parameters, ensure_dir, make_noise, save_generated_grid


def parse_args() -> argparse.Namespace:
    """解析 DCGAN 训练所需的命令行参数。"""

    default_config = PROJECT_ROOT / "src" / "gan_faces" / "config" / "dcgan.yaml"
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=str(default_config), help="YAML 配置文件路径")
    config_args, remaining_argv = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(description="训练 DCGAN 人头图像生成模型")
    parser.add_argument("--config", type=str, default=str(default_config), help="YAML 配置文件路径")
    parser.add_argument("--dataset", choices=["folder", "lfw", "celeba"], default="folder")
    parser.add_argument("--data-root", type=str, default="data/faces")
    parser.add_argument("--download", action="store_true", help="允许 torchvision 下载数据集")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128, help="总 batch size，多卡时会自动按卡均分")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--generator-features", type=int, default=64)
    parser.add_argument("--discriminator-features", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--kimg-per-log", type=float, default=0.0, help="按 kimg 触发日志，<=0 时退回 log-every")
    parser.add_argument("--sample-every-kimg", type=float, default=1.0, help="每隔多少 kimg 保存一次样例图，<=0 时退回 sample-every")
    parser.add_argument("--save-every-kimg", type=float, default=5.0, help="每隔多少 kimg 保存一次 checkpoint，<=0 时退回 save-every")
    parser.add_argument("--eval-every-kimg", type=float, default=0.0, help="每隔多少 kimg 做一次中间评测，<=0 时关闭")
    parser.add_argument("--metrics", type=str, default="", help="中间评测指标，逗号分隔，如 fid5k,is5k；为空则关闭")
    parser.add_argument("--eval-data-root", type=str, default="", help="中间评测使用的真实数据路径，默认与 data-root 相同")
    parser.add_argument("--eval-verbose", action="store_true", help="打印中间评测的详细进度")
    parser.add_argument("--no-eval-cache", action="store_true", help="禁用中间评测的真实特征缓存")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="生成器 EMA 衰减系数，<=0 时关闭")
    parser.add_argument("--gpus", type=int, default=1, help="参与训练的 GPU 数量")
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=str, default="")
    parser.add_argument("--tensorboard-dir", type=str, default="", help="TensorBoard 日志目录，默认 output-dir/tensorboard")
    parser.add_argument("--no-tensorboard", action="store_true", help="关闭 TensorBoard 日志写入")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    apply_config_defaults(parser, load_yaml_config(config_args.config))
    return parser.parse_args(remaining_argv)


def build_training_components(args: argparse.Namespace, device: torch.device):
    """构建当前脚本用到的模型、优化器和损失函数。"""

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
    return generator, discriminator, criterion, optimizer_g, optimizer_d, model_args, disc_args


def _next_kimg_boundary(cur_nimg: int, interval_kimg: float) -> int | None:
    """根据当前 nimg 计算下一次按 kimg 触发的边界。"""

    if interval_kimg <= 0:
        return None
    interval_nimg = max(int(round(interval_kimg * 1000)), 1)
    return ((cur_nimg // interval_nimg) + 1) * interval_nimg


def train_worker(rank: int, args: argparse.Namespace) -> None:
    """单个训练子进程：单卡时直接运行，多卡时由 spawn 启动。"""

    world_size = args.gpus
    device = prepare_device(args.device, rank, world_size)
    setup_distributed(rank, world_size, device, args.master_addr, args.master_port)
    seed_everything(args.seed + rank)
    torch.backends.cudnn.benchmark = device.type == "cuda"
    install_signal_handlers()

    logger = None
    layout = None
    dataloader = None
    dataset = None
    sampler = None
    generator = None
    discriminator = None
    generator_ema = None
    criterion = None
    optimizer_g = None
    optimizer_d = None
    fixed_noise = None
    is_rank0 = rank == 0

    try:
        if args.image_size != 64:
            raise ValueError("当前 DCGAN 结构固定输出 64x64 图片，请保持 --image-size 64")
        if args.batch_size % world_size != 0:
            raise ValueError(f"--batch-size {args.batch_size} 不能被 --gpus {world_size} 整除")
        requested_metrics = parse_metrics(args.metrics) if args.metrics.strip() else []

        if is_rank0:
            layout = create_training_layout(args.output_dir, args.tensorboard_dir)
            runtime_options = {**vars(args), "world_size": world_size}
            save_training_options(layout, runtime_options)
            logger = TrainingLogger(layout, enable_tensorboard=not args.no_tensorboard)
            logger.log_config(runtime_options)
            print(f"输出目录: {layout.output_dir}")
            if not args.no_tensorboard:
                print(f"TensorBoard 日志目录: {layout.tensorboard_dir}")
                print(f"查看命令: tensorboard --logdir {layout.tensorboard_dir}")
        barrier()

        dataset = build_dataset(args.dataset, args.data_root, args.image_size, download=args.download)
        per_rank_batch_size = args.batch_size // world_size
        drop_last = len(dataset) >= args.batch_size
        sampler = None
        if world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=drop_last,
            )
        dataloader = build_dataloader(
            dataset,
            per_rank_batch_size,
            args.workers,
            shuffle=sampler is None,
            drop_last=drop_last,
            sampler=sampler,
        )

        if is_rank0:
            print(
                f"数据集: {args.dataset}, 图片数量: {len(dataset)}, 设备: {device}, "
                f"总 batch: {args.batch_size}, 每卡 batch: {per_rank_batch_size}, GPU 数: {world_size}"
            )

        generator, discriminator, criterion, optimizer_g, optimizer_d, model_args, disc_args = build_training_components(args, device)
        generator_ema = copy.deepcopy(generator).eval() if is_rank0 and args.ema_decay > 0 else None
        if is_rank0:
            print(
                f"参数量: G={count_parameters(generator):,}, D={count_parameters(discriminator):,}"
            )

        start_epoch = 1
        cur_nimg = 0
        if args.resume:
            checkpoint = torch.load(args.resume, map_location=device)
            generator.load_state_dict(checkpoint["generator"])
            discriminator.load_state_dict(checkpoint["discriminator"])
            optimizer_g.load_state_dict(checkpoint["optimizer_g"])
            optimizer_d.load_state_dict(checkpoint["optimizer_d"])
            if generator_ema is not None:
                generator_ema.load_state_dict(checkpoint.get("generator_ema", checkpoint["generator"]))
            start_epoch = int(checkpoint["epoch"]) + 1
            cur_nimg = int(checkpoint.get("cur_nimg", 0))
            if is_rank0:
                print(f"已从 {args.resume} 恢复训练，将从 epoch {start_epoch} 开始")

        generator = maybe_wrap_ddp(generator, device, world_size)
        discriminator = maybe_wrap_ddp(discriminator, device, world_size)

        fixed_noise = make_noise(min(64, args.batch_size), args.latent_dim, device) if is_rank0 else None
        saved_reals = False
        total_steps = len(dataloader)
        next_sample_nimg = _next_kimg_boundary(cur_nimg, args.sample_every_kimg)
        next_save_nimg = _next_kimg_boundary(cur_nimg, args.save_every_kimg)
        next_eval_nimg = _next_kimg_boundary(cur_nimg, args.eval_every_kimg)
        next_log_nimg = _next_kimg_boundary(cur_nimg, args.kimg_per_log)

        if is_rank0 and fixed_noise is not None:
            with torch.no_grad():
                init_samples = (generator_ema or unwrap_module(generator)).eval()(fixed_noise)
            logger.save_samples(init_samples, layout.sample_dir / "fakes_init.png", global_step=0, nrow=8)

        for epoch in range(start_epoch, args.epochs + 1):
            if interrupt_requested():
                raise KeyboardInterrupt
            epoch_start = time.time()
            if sampler is not None:
                sampler.set_epoch(epoch)

            unwrap_module(generator).train()
            unwrap_module(discriminator).train()
            epoch_sums = {"loss_d": 0.0, "loss_g": 0.0, "d_real": 0.0, "d_fake": 0.0}

            for step, real_images in enumerate(dataloader, start=1):
                if interrupt_requested():
                    raise KeyboardInterrupt
                step_start = time.perf_counter()
                real_images = real_images.to(device, non_blocking=True)
                batch_size = real_images.size(0)

                if is_rank0 and not saved_reals:
                    save_generated_grid(real_images, layout.sample_dir / "reals.png", nrow=8)
                    saved_reals = True

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

                optimizer_g.zero_grad(set_to_none=True)
                fool_targets = torch.ones(batch_size, device=device)
                fake_scores_for_g = discriminator(fake_images)
                loss_g = criterion(fake_scores_for_g, fool_targets)
                loss_g.backward()
                optimizer_g.step()

                if generator_ema is not None:
                    update_ema(generator_ema, generator, args.ema_decay)

                metric_tensor = torch.stack(
                    [
                        loss_d.detach(),
                        loss_g.detach(),
                        real_scores.detach().mean(),
                        fake_scores.detach().mean(),
                    ]
                )
                metric_tensor = average_tensor(metric_tensor, world_size)
                metrics = {
                    "loss_d": metric_tensor[0].item(),
                    "loss_g": metric_tensor[1].item(),
                    "d_real": metric_tensor[2].item(),
                    "d_fake": metric_tensor[3].item(),
                }
                global_batch_size = batch_size * world_size
                prev_nimg = cur_nimg
                cur_nimg += global_batch_size
                cur_kimg = cur_nimg / 1000.0

                for key in epoch_sums:
                    epoch_sums[key] += metrics[key]

                if is_rank0:
                    global_step = (epoch - 1) * total_steps + step
                    sec_per_step = time.perf_counter() - step_start
                    images_seen = float(cur_nimg)
                    extras = {
                        "Progress/epoch": float(epoch),
                        "Progress/step": float(step),
                        "Progress/images_seen": images_seen,
                        "Progress/nimg": float(cur_nimg),
                        "Progress/kimg": cur_kimg,
                        "Timing/sec_per_step": sec_per_step,
                        "Timing/images_per_sec": global_batch_size / max(sec_per_step, 1e-8),
                    }
                    if device.type == "cuda":
                        extras["Resources/gpu_mem_allocated_gb"] = torch.cuda.memory_allocated(device) / (2**30)
                        extras["Resources/gpu_mem_reserved_gb"] = torch.cuda.memory_reserved(device) / (2**30)
                    should_log = step == 1 or step == total_steps
                    if args.kimg_per_log > 0:
                        if next_log_nimg is not None and cur_nimg >= next_log_nimg:
                            should_log = True
                            next_log_nimg = _next_kimg_boundary(cur_nimg, args.kimg_per_log)
                    elif step % args.log_every == 0:
                        should_log = True
                    if should_log:
                        logger.log_step(
                            epoch=epoch,
                            step=step,
                            total_steps=total_steps,
                            global_step=global_step,
                            metrics=metrics,
                            lr_g=optimizer_g.param_groups[0]["lr"],
                            lr_d=optimizer_d.param_groups[0]["lr"],
                            extras=extras,
                        )
                        print(
                            f"Epoch [{epoch}/{args.epochs}] Step [{step}/{total_steps}] "
                            f"nimg={cur_nimg} kimg={cur_kimg:.3f} "
                            f"Loss_D={metrics['loss_d']:.4f} Loss_G={metrics['loss_g']:.4f} "
                            f"D(real)={metrics['d_real']:.4f} D(fake)={metrics['d_fake']:.4f}"
                        )

                    should_sample = False
                    if args.sample_every_kimg > 0:
                        if next_sample_nimg is not None and cur_nimg >= next_sample_nimg:
                            should_sample = True
                            next_sample_nimg = _next_kimg_boundary(cur_nimg, args.sample_every_kimg)
                    elif epoch % args.sample_every == 0 and step == total_steps:
                        should_sample = True

                    should_save = False
                    if args.save_every_kimg > 0:
                        if next_save_nimg is not None and cur_nimg >= next_save_nimg:
                            should_save = True
                            next_save_nimg = _next_kimg_boundary(cur_nimg, args.save_every_kimg)
                    elif epoch % args.save_every == 0 and step == total_steps:
                        should_save = True

                    if should_sample:
                        sample_model = generator_ema if generator_ema is not None else unwrap_module(generator).eval()
                        with torch.no_grad():
                            samples = sample_model(fixed_noise)
                        logger.save_samples(
                            samples,
                            layout.sample_dir / f"fakes{cur_nimg // 1000:06d}.png",
                            global_step=global_step,
                            nrow=8,
                        )

                    state = {
                        "model_type": "dcgan",
                        "epoch": epoch,
                        "cur_nimg": cur_nimg,
                        "generator": unwrap_module(generator).state_dict(),
                        "generator_ema": generator_ema.state_dict() if generator_ema is not None else unwrap_module(generator).state_dict(),
                        "discriminator": unwrap_module(discriminator).state_dict(),
                        "optimizer_g": optimizer_g.state_dict(),
                        "optimizer_d": optimizer_d.state_dict(),
                        "model_args": model_args,
                        "disc_args": disc_args,
                        "train_args": vars(args),
                        "world_size": world_size,
                    }
                    if should_save or step == total_steps:
                        torch.save(state, layout.checkpoint_dir / "latest.pt")
                    if should_save:
                        torch.save(state, layout.checkpoint_dir / f"dcgan_nimg_{cur_nimg:08d}.pt")

                    should_eval = False
                    if requested_metrics and args.eval_every_kimg > 0:
                        if next_eval_nimg is not None and cur_nimg >= next_eval_nimg:
                            should_eval = True
                            next_eval_nimg = _next_kimg_boundary(cur_nimg, args.eval_every_kimg)
                    if should_eval:
                        eval_model = generator_ema if generator_ema is not None else unwrap_module(generator).eval()
                        adapter = AI2602GeneratorAdapter.from_generator(eval_model, z_dim=args.latent_dim, device=device)
                        adapter.eval().requires_grad_(False).to(device)
                        eval_result = evaluate_adapter(
                            adapter=adapter,
                            data_path=args.eval_data_root or args.data_root,
                            metrics=requested_metrics,
                            verbose=args.eval_verbose,
                            cache=not args.no_eval_cache,
                        )
                        metric_scalars = {
                            f"Metrics/{metric_name}/{key}": float(value)
                            for metric_name, values in eval_result["metrics"].items()
                            for key, value in values.items()
                        }
                        logger.log_epoch(
                            epoch=epoch,
                            global_step=global_step,
                            metrics={},
                            elapsed_sec=0.0,
                            extras={
                                "Progress/nimg": float(cur_nimg),
                                "Progress/kimg": cur_kimg,
                                **metric_scalars,
                            },
                        )
                        print(f"中间评测 @ nimg={cur_nimg}, kimg={cur_kimg:.3f}: {eval_result['metrics']}")

            if is_rank0:
                epoch_metrics = {key: value / max(total_steps, 1) for key, value in epoch_sums.items()}
                global_step = epoch * total_steps
                logger.log_epoch(
                    epoch=epoch,
                    global_step=global_step,
                    metrics=epoch_metrics,
                    elapsed_sec=time.time() - epoch_start,
                    extras={
                        "Progress/images_seen": float(cur_nimg),
                        "Progress/nimg": float(cur_nimg),
                        "Progress/kimg": cur_nimg / 1000.0,
                        "Timing/sec_per_epoch": time.time() - epoch_start,
                    },
                )

        if is_rank0:
            print("DCGAN 训练完成。")
    except KeyboardInterrupt:
        if is_rank0:
            print("\n收到中断信号，正在清理训练进程并释放资源...")
    finally:
        cleanup_training_process(
            logger=logger,
            dataloader=dataloader,
            device=device,
            modules=[
                generator,
                discriminator,
                generator_ema,
                criterion,
                optimizer_g,
                optimizer_d,
                fixed_noise,
                sampler,
                dataset,
            ],
        )
        restore_signal_handlers()


def main() -> None:
    """单卡直接运行，多卡用 spawn 拉起多个训练进程。"""

    args = parse_args()
    args.output_dir = str(ensure_dir(args.output_dir))
    if args.gpus > 1:
        if args.master_port == "":
            args.master_port = find_free_port()
        torch.multiprocessing.set_start_method("spawn", force=True)
        process_context = None
        try:
            process_context = torch.multiprocessing.spawn(train_worker, args=(args,), nprocs=args.gpus, join=False)
            while not process_context.join(timeout=1.0):
                pass
        except KeyboardInterrupt:
            print("\n主进程收到中断，正在停止所有训练子进程...")
            if process_context is not None:
                for process in process_context.processes:
                    if process.is_alive():
                        process.terminate()
                for process in process_context.processes:
                    process.join(timeout=5.0)
            cleanup_distributed()
    else:
        if args.master_port == "":
            args.master_port = "29500"
        train_worker(rank=0, args=args)


if __name__ == "__main__":
    main()
