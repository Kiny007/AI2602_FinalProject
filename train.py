"""DCGAN 训练入口。

当前入口采用更偏 NVIDIA 风格的训练组织方式：以 `kimg` 作为统一进度单位，
并使用 accelerate 管理单卡/多卡与混合精度。
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from collections import OrderedDict
from datetime import timedelta
from pathlib import Path

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import InitProcessGroupKwargs
from torch import nn, optim

# Unrolled GAN 需要函数式调用判别器，以便临时更新 D 的参数而不改动真实模型。
try:
    from torch.func import functional_call
except ImportError:  # pragma: no cover - compatibility with older torch.
    from torch.nn.utils.stateless import functional_call


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gan_faces.data import build_dataloader, build_dataset
from gan_faces.models import Discriminator, Generator, init_dcgan_weights
from gan_faces.config_utils import apply_config_defaults, load_yaml_config
from gan_faces.eval.evaluate import AI2602GeneratorAdapter
from gan_faces.eval.nvidia_evaluator import evaluate_adapter, parse_metrics
from gan_faces.train_runtime import (
    TrainingLogger,
    cleanup_training_process,
    create_training_layout,
    install_signal_handlers,
    interrupt_requested,
    restore_signal_handlers,
    save_training_options,
    seed_everything,
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
    parser.add_argument("--batch-size", type=int, default=128, help="总 batch size，多卡时会自动按卡均分")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--generator-features", type=int, default=64)
    parser.add_argument("--discriminator-features", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    # Unrolled GAN 相关开关：n_dis=1 保持 DCGAN 的 1:1 更新节奏；
    # unroll_steps>0 时，G 会基于可微分展开后的 D 来更新。
    parser.add_argument("--n-dis", type=int, default=1, help="Discriminator updates per generator update")
    parser.add_argument("--unroll-steps", type=int, default=0, help="Differentiable discriminator steps for Unrolled GAN; 0 keeps vanilla DCGAN")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--total-kimg", type=float, default=1000.0, help="总训练长度，单位 kimg")
    parser.add_argument("--kimg-per-log", type=float, default=0.5, help="每隔多少 kimg 打一次日志")
    parser.add_argument("--sample-every-kimg", type=float, default=1.0, help="每隔多少 kimg 保存一次样例图")
    parser.add_argument("--save-every-kimg", type=float, default=5.0, help="每隔多少 kimg 保存一次 checkpoint")
    parser.add_argument("--eval-every-kimg", type=float, default=0.0, help="每隔多少 kimg 做一次中间评测，<=0 时关闭")
    parser.add_argument("--metrics", type=str, default="", help="中间评测指标，逗号分隔，如 fid5k,is5k；为空则关闭")
    parser.add_argument("--eval-data-root", type=str, default="", help="中间评测使用的真实数据路径，默认与 data-root 相同")
    parser.add_argument("--eval-verbose", action="store_true", help="打印中间评测的详细进度")
    parser.add_argument("--no-eval-cache", action="store_true", help="禁用中间评测的真实特征缓存")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="生成器 EMA 衰减系数，<=0 时关闭")
    parser.add_argument("--tensorboard-dir", type=str, default="", help="TensorBoard 日志目录，默认 output-dir/tensorboard")
    parser.add_argument("--no-tensorboard", action="store_true", help="关闭 TensorBoard 日志写入")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    # 将 fp16 作为脚本默认混合精度，避免完全依赖外部 accelerate 配置。
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="fp16", help="accelerate mixed precision mode")
    parser.add_argument("--allow-tf32", action="store_true", help="允许 CUDA matmul/conv 使用 TF32 提升吞吐")
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

    criterion = nn.BCEWithLogitsLoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    return generator, discriminator, criterion, optimizer_g, optimizer_d, model_args, disc_args


def _optimizer_core(optimizer):
    """取出 accelerate 包装前的 torch optimizer。"""

    return getattr(optimizer, "optimizer", optimizer)


def _state_step_value(raw_step) -> float:
    """把 Adam step 计数统一成 Python float，便于可微分展开计算。"""

    if raw_step is None:
        return 0.0
    if torch.is_tensor(raw_step):
        return float(raw_step.detach().item())
    return float(raw_step)


def _prepare_unrolled_adam_state(param_items, optimizer):
    """复制 Adam 状态，让临时 D 更新贴近真实训练但不污染 optimizer。"""

    optimizer_core = _optimizer_core(optimizer)
    param_groups = optimizer_core.param_groups
    state_store = optimizer_core.state
    default_group = param_groups[0]
    group_by_param = {
        id(param): group
        for group in param_groups
        for param in group["params"]
    }
    unrolled_state = {}
    for name, param in param_items:
        group = group_by_param.get(id(param), default_group)
        state = state_store.get(param, {})
        exp_avg = state.get("exp_avg")
        exp_avg_sq = state.get("exp_avg_sq")
        max_exp_avg_sq = state.get("max_exp_avg_sq")
        unrolled_state[name] = {
            "group": group,
            "step": _state_step_value(state.get("step")),
            "exp_avg": (
                torch.zeros_like(param, memory_format=torch.preserve_format)
                if exp_avg is None
                else exp_avg.detach().to(device=param.device, dtype=param.dtype)
            ),
            "exp_avg_sq": (
                torch.zeros_like(param, memory_format=torch.preserve_format)
                if exp_avg_sq is None
                else exp_avg_sq.detach().to(device=param.device, dtype=param.dtype)
            ),
            "max_exp_avg_sq": (
                None
                if max_exp_avg_sq is None
                else max_exp_avg_sq.detach().to(device=param.device, dtype=param.dtype)
            ),
        }
    return unrolled_state


def _adam_unroll_step(params, grads, unrolled_state):
    """对临时判别器参数执行一步可微分 Adam 更新。"""

    next_params = OrderedDict()
    next_state = {}
    for (name, param), grad in zip(params.items(), grads):
        state = unrolled_state[name]
        group = state["group"]
        beta1, beta2 = group["betas"]
        step = state["step"] + 1.0
        if grad is None:
            grad = torch.zeros_like(param, memory_format=torch.preserve_format)
        if group.get("maximize", False):
            grad = -grad
        weight_decay = group.get("weight_decay", 0.0)
        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        exp_avg = state["exp_avg"] * beta1 + grad * (1.0 - beta1)
        exp_avg_sq = state["exp_avg_sq"] * beta2 + grad.square() * (1.0 - beta2)
        if group.get("amsgrad", False):
            prev_max = state["max_exp_avg_sq"]
            if prev_max is None:
                prev_max = torch.zeros_like(param, memory_format=torch.preserve_format)
            max_exp_avg_sq = torch.maximum(prev_max, exp_avg_sq)
            denom_source = max_exp_avg_sq
        else:
            max_exp_avg_sq = state["max_exp_avg_sq"]
            denom_source = exp_avg_sq

        bias_correction1 = 1.0 - beta1**step
        bias_correction2 = 1.0 - beta2**step
        step_size = group["lr"] / bias_correction1
        denom = denom_source.sqrt() / (bias_correction2**0.5)
        denom = denom.add(group.get("eps", 1e-8))
        next_params[name] = param.addcdiv(exp_avg, denom, value=-step_size)
        next_state[name] = {
            "group": group,
            "step": step,
            "exp_avg": exp_avg,
            "exp_avg_sq": exp_avg_sq,
            "max_exp_avg_sq": max_exp_avg_sq,
        }
    return next_params, next_state


def _functional_discriminator(discriminator, params, buffers, images):
    """用临时参数和复制的 buffer 运行判别器。"""

    state = OrderedDict(params)
    state.update(buffers)
    return functional_call(discriminator, state, (images,))


def _unrolled_generator_loss(
    generator,
    discriminator,
    criterion,
    optimizer_d,
    real_images,
    latent_dim: int,
    device: torch.device,
    unroll_steps: int,
    real_targets,
    fake_targets,
    fool_targets,
):
    """模拟若干步 D 更新后，再计算生成器损失。"""

    param_items = tuple(discriminator.named_parameters())
    params = OrderedDict(param_items)
    buffers = OrderedDict(
        (name, buffer.detach().clone())
        for name, buffer in discriminator.named_buffers()
    )
    unrolled_state = _prepare_unrolled_adam_state(param_items, optimizer_d)

    noise = make_noise(real_images.size(0), latent_dim, device)
    fake_images = generator(noise)
    for _ in range(unroll_steps):
        # 内层 D 更新属于 G 的计算图，因此 fake_images 不能 detach。
        real_scores = _functional_discriminator(discriminator, params, buffers, real_images)
        fake_scores = _functional_discriminator(discriminator, params, buffers, fake_images)
        loss_d = criterion(real_scores.float(), real_targets.float()) + criterion(fake_scores.float(), fake_targets.float())
        grads = torch.autograd.grad(
            loss_d,
            tuple(params.values()),
            # 保留计算图，使 G 能通过“模拟后的 D 更新”收到梯度。
            create_graph=True,
            allow_unused=True,
        )
        params, unrolled_state = _adam_unroll_step(params, grads, unrolled_state)

    fake_scores_for_g = _functional_discriminator(discriminator, params, buffers, fake_images)
    loss_g = criterion(fake_scores_for_g.float(), fool_targets.float())
    return loss_g, fake_scores_for_g


def _next_kimg_boundary(cur_nimg: int, interval_kimg: float) -> int | None:
    """根据当前 nimg 计算下一次按 kimg 触发的边界。"""

    if interval_kimg <= 0:
        return None
    interval_nimg = max(int(round(interval_kimg * 1000)), 1)
    return ((cur_nimg // interval_nimg) + 1) * interval_nimg


def _format_duration(seconds: float) -> str:
    """把秒数格式化成更适合日志阅读的时长字符串。"""

    total_seconds = max(int(seconds), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    return f"{minutes:d}m {secs:02d}s"


def train(args: argparse.Namespace) -> None:
    """使用 accelerate 统一单卡、多卡与混合精度训练。"""

    ddp_kwargs = DistributedDataParallelKwargs(broadcast_buffers=False)
    init_pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=60))
    accelerator = Accelerator(
        gradient_accumulation_steps=max(args.gradient_accumulation_steps, 1),
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[ddp_kwargs, init_pg_kwargs],
    )
    device = accelerator.device
    world_size = accelerator.num_processes
    seed_everything(args.seed + accelerator.process_index)
    torch.backends.cudnn.benchmark = device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.allow_tf32 = args.allow_tf32
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
    scaler_state = None
    is_rank0 = accelerator.is_main_process

    try:
        if args.image_size != 64:
            raise ValueError("当前 DCGAN 结构固定输出 64x64 图片，请保持 --image-size 64")
        if args.batch_size % world_size != 0:
            raise ValueError(f"--batch-size {args.batch_size} 不能被当前进程数 {world_size} 整除")
        if args.n_dis < 1:
            raise ValueError("--n-dis must be >= 1")
        if args.unroll_steps < 0:
            raise ValueError("--unroll-steps must be >= 0")
        if args.total_kimg <= 0:
            raise ValueError("--total-kimg 必须为正数")
        if args.kimg_per_log <= 0:
            raise ValueError("--kimg-per-log 必须为正数")
        if args.sample_every_kimg <= 0:
            raise ValueError("--sample-every-kimg 必须为正数")
        if args.save_every_kimg <= 0:
            raise ValueError("--save-every-kimg 必须为正数")
        requested_metrics = parse_metrics(args.metrics) if args.metrics.strip() else []

        if is_rank0:
            layout = create_training_layout(args.output_dir, args.tensorboard_dir)
            runtime_options = {
                **vars(args),
                "world_size": world_size,
                "mixed_precision_runtime": accelerator.mixed_precision,
            }
            save_training_options(layout, runtime_options)
            logger = TrainingLogger(layout, enable_tensorboard=not args.no_tensorboard)
            logger.log_config(runtime_options)
            print(f"输出目录: {layout.output_dir}")
            if not args.no_tensorboard:
                print(f"TensorBoard 日志目录: {layout.tensorboard_dir}")
                print(f"查看命令: tensorboard --logdir {layout.tensorboard_dir}")
        accelerator.wait_for_everyone()

        dataset = build_dataset(args.dataset, args.data_root, args.image_size, download=args.download)
        per_rank_batch_size = args.batch_size // world_size
        drop_last = len(dataset) >= args.batch_size
        sampler = None
        dataloader = build_dataloader(
            dataset,
            per_rank_batch_size,
            args.workers,
            shuffle=True,
            drop_last=drop_last,
            sampler=None,
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

        cur_nimg = 0
        current_epoch = 1
        global_step = int(cur_nimg // 1000)
        if args.resume:
            checkpoint = torch.load(args.resume, map_location="cpu")
            generator.load_state_dict(checkpoint["generator"])
            discriminator.load_state_dict(checkpoint["discriminator"])
            optimizer_g.load_state_dict(checkpoint["optimizer_g"])
            optimizer_d.load_state_dict(checkpoint["optimizer_d"])
            scaler_state = checkpoint.get("scaler")
            if generator_ema is not None and "generator_ema" in checkpoint:
                generator_ema.load_state_dict(checkpoint["generator_ema"])
            current_epoch = int(checkpoint.get("epoch", 0)) + 1
            cur_nimg = int(checkpoint.get("cur_nimg", 0))
            global_step = int(checkpoint.get("global_step", cur_nimg // 1000))
            if is_rank0:
                print(f"已从 {args.resume} 恢复训练，将从 nimg={cur_nimg} 继续")

        generator, discriminator, optimizer_g, optimizer_d, dataloader = accelerator.prepare(
            generator,
            discriminator,
            optimizer_g,
            optimizer_d,
            dataloader,
        )
        if scaler_state is not None and getattr(accelerator, "scaler", None) is not None:
            accelerator.scaler.load_state_dict(scaler_state)

        fixed_noise = make_noise(min(64, args.batch_size), args.latent_dim, device) if is_rank0 else None
        saved_reals = False
        total_steps = len(dataloader)
        train_start_time = time.time()
        next_sample_nimg = _next_kimg_boundary(cur_nimg, args.sample_every_kimg)
        next_save_nimg = _next_kimg_boundary(cur_nimg, args.save_every_kimg)
        next_eval_nimg = _next_kimg_boundary(cur_nimg, args.eval_every_kimg)
        next_log_nimg = _next_kimg_boundary(cur_nimg, args.kimg_per_log)
        total_nimg = max(int(round(args.total_kimg * 1000)), 1)

        if is_rank0 and fixed_noise is not None:
            with torch.no_grad():
                init_model = generator_ema if generator_ema is not None else accelerator.unwrap_model(generator).eval()
                init_samples = init_model(fixed_noise)
            logger.save_samples(init_samples, layout.sample_dir / "fakes_init.png", global_step=0, nrow=8)
        if world_size > 1:
            accelerator.wait_for_everyone()

        while cur_nimg < total_nimg:
            if interrupt_requested():
                raise KeyboardInterrupt
            current_epoch_start = time.time()

            generator.train()
            discriminator.train()

            for step, real_images in enumerate(dataloader, start=1):
                if interrupt_requested():
                    raise KeyboardInterrupt
                step_start = time.perf_counter()
                real_images = real_images.to(device, non_blocking=True)
                batch_size = real_images.size(0)

                if not saved_reals:
                    if is_rank0:
                        save_generated_grid(real_images, layout.sample_dir / "reals.png", nrow=8)
                    saved_reals = True
                    if world_size > 1:
                        accelerator.wait_for_everyone()

                real_targets = torch.ones(batch_size, device=device)
                fake_targets = torch.zeros(batch_size, device=device)

                # 真实判别器更新：这里会像普通 DCGAN 一样实际修改 D。
                for _ in range(args.n_dis):
                    with accelerator.accumulate(discriminator):
                        optimizer_d.zero_grad(set_to_none=True)
                        with accelerator.autocast():
                            real_scores = discriminator(real_images)
                            loss_d_real = criterion(real_scores.float(), real_targets.float())

                            noise = make_noise(batch_size, args.latent_dim, device)
                            with torch.no_grad():
                                fake_images_d = generator(noise)
                            fake_scores = discriminator(fake_images_d)
                            loss_d_fake = criterion(fake_scores.float(), fake_targets.float())
                            loss_d = loss_d_real + loss_d_fake
                        accelerator.backward(loss_d)
                        optimizer_d.step()

                with accelerator.accumulate(generator):
                    optimizer_g.zero_grad(set_to_none=True)
                    fool_targets = torch.ones(batch_size, device=device)
                    with accelerator.autocast():
                        if args.unroll_steps > 0:
                            # Unrolled GAN 分支：先临时模拟 D 更新若干步，再用这个 D 更新 G。
                            loss_g, fake_scores_for_g = _unrolled_generator_loss(
                                generator=generator,
                                discriminator=accelerator.unwrap_model(discriminator),
                                criterion=criterion,
                                optimizer_d=optimizer_d,
                                real_images=real_images,
                                latent_dim=args.latent_dim,
                                device=device,
                                unroll_steps=args.unroll_steps,
                                real_targets=real_targets,
                                fake_targets=fake_targets,
                                fool_targets=fool_targets,
                            )
                        else:
                            noise = make_noise(batch_size, args.latent_dim, device)
                            fake_images_g = generator(noise)
                            fake_scores_for_g = discriminator(fake_images_g)
                            loss_g = criterion(fake_scores_for_g.float(), fool_targets.float())
                    accelerator.backward(loss_g)
                    optimizer_g.step()
                    # unroll 分支可能留下临时 D 梯度，下一轮真实 D 更新前先清空。
                    optimizer_d.zero_grad(set_to_none=True)

                if generator_ema is not None:
                    update_ema(generator_ema, generator, args.ema_decay)

                metric_tensor = torch.stack(
                    [
                        loss_d.detach(),
                        loss_g.detach(),
                        torch.sigmoid(real_scores.detach()).mean(),
                        torch.sigmoid(fake_scores.detach()).mean(),
                    ]
                )
                metric_tensor = accelerator.reduce(metric_tensor, reduction="mean")
                metrics = {
                    "loss_d": metric_tensor[0].item(),
                    "loss_g": metric_tensor[1].item(),
                    "d_real": metric_tensor[2].item(),
                    "d_fake": metric_tensor[3].item(),
                }
                global_batch_size = batch_size * world_size
                cur_nimg += global_batch_size
                global_step = int(cur_nimg // 1000)
                cur_kimg = cur_nimg / 1000.0
                reached_end = cur_nimg >= total_nimg
                should_eval = False
                if requested_metrics and args.eval_every_kimg > 0:
                    should_eval = reached_end or (next_eval_nimg is not None and cur_nimg >= next_eval_nimg)
                    if should_eval and next_eval_nimg is not None and cur_nimg >= next_eval_nimg:
                        next_eval_nimg = _next_kimg_boundary(cur_nimg, args.eval_every_kimg)

                if should_eval and world_size > 1:
                    accelerator.wait_for_everyone()

                if is_rank0:
                    sec_per_step = time.perf_counter() - step_start
                    sec_per_kimg = sec_per_step * 1000.0 / max(global_batch_size, 1)
                    elapsed_time = time.time() - train_start_time
                    images_seen = float(cur_nimg)
                    extras = {
                        "Progress/epoch": float(current_epoch),
                        "Progress/images_seen": images_seen,
                        "Progress/nimg": float(cur_nimg),
                        "Progress/kimg": cur_kimg,
                        "Timing/elapsed_sec": elapsed_time,
                        "Timing/sec_per_step": sec_per_step,
                        "Timing/sec_per_kimg": sec_per_kimg,
                        "Timing/images_per_sec": global_batch_size / max(sec_per_step, 1e-8),
                    }
                    if device.type == "cuda":
                        extras["Resources/gpu_mem_allocated_gb"] = torch.cuda.memory_allocated(device) / (2**30)
                        extras["Resources/gpu_mem_reserved_gb"] = torch.cuda.memory_reserved(device) / (2**30)
                    should_log = reached_end or (next_log_nimg is not None and cur_nimg >= next_log_nimg)
                    if should_log:
                        if next_log_nimg is not None and cur_nimg >= next_log_nimg:
                            next_log_nimg = _next_kimg_boundary(cur_nimg, args.kimg_per_log)
                        logger.log_step(
                            epoch=current_epoch,
                            step=step,
                            total_steps=total_steps,
                            global_step=global_step,
                            metrics=metrics,
                            lr_g=optimizer_g.param_groups[0]["lr"],
                            lr_d=optimizer_d.param_groups[0]["lr"],
                            extras=extras,
                        )
                        print(
                            f"Epoch {current_epoch} Step [{step}/{total_steps}] "
                            f"kimg={cur_kimg:.3f} time={_format_duration(elapsed_time)} "
                            f"sec/kimg={sec_per_kimg:.2f} "
                            f"Loss_D={metrics['loss_d']:.4f} Loss_G={metrics['loss_g']:.4f} "
                            f"D(real)={metrics['d_real']:.4f} D(fake)={metrics['d_fake']:.4f}"
                        )

                    should_sample = reached_end or (next_sample_nimg is not None and cur_nimg >= next_sample_nimg)
                    if should_sample and next_sample_nimg is not None and cur_nimg >= next_sample_nimg:
                        next_sample_nimg = _next_kimg_boundary(cur_nimg, args.sample_every_kimg)

                    should_save = reached_end or (next_save_nimg is not None and cur_nimg >= next_save_nimg)
                    if should_save and next_save_nimg is not None and cur_nimg >= next_save_nimg:
                        next_save_nimg = _next_kimg_boundary(cur_nimg, args.save_every_kimg)

                    if should_sample:
                        sample_model = generator_ema if generator_ema is not None else accelerator.unwrap_model(generator).eval()
                        with torch.no_grad():
                            samples = sample_model(fixed_noise)
                        logger.save_samples(
                            samples,
                            layout.sample_dir / f"fakes{cur_nimg // 1000:06d}.png",
                            global_step=global_step,
                            nrow=8,
                        )

                    # 区分保存类型，方便后续对比 DCGAN 与 Unrolled DCGAN。
                    model_type = "unrolled_dcgan" if args.unroll_steps > 0 else "dcgan"
                    state = {
                        "model_type": model_type,
                        "epoch": current_epoch,
                        "cur_nimg": cur_nimg,
                        "global_step": global_step,
                        "generator": accelerator.get_state_dict(generator),
                        "generator_ema": generator_ema.state_dict() if generator_ema is not None else accelerator.get_state_dict(generator),
                        "discriminator": accelerator.get_state_dict(discriminator),
                        "optimizer_g": optimizer_g.state_dict(),
                        "optimizer_d": optimizer_d.state_dict(),
                        "scaler": accelerator.scaler.state_dict() if getattr(accelerator, "scaler", None) is not None else None,
                        "model_args": model_args,
                        "disc_args": disc_args,
                        "train_args": vars(args),
                        "world_size": world_size,
                        "mixed_precision": accelerator.mixed_precision,
                    }
                    if should_save:
                        torch.save(state, layout.checkpoint_dir / "latest.pt")
                        torch.save(state, layout.checkpoint_dir / f"{model_type}_nimg_{cur_nimg:08d}.pt")

                    if should_eval:
                        eval_model = generator_ema if generator_ema is not None else accelerator.unwrap_model(generator).eval()
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
                        logger.log_step(
                            epoch=current_epoch,
                            step=step,
                            total_steps=total_steps,
                            global_step=global_step,
                            metrics=metrics,
                            lr_g=optimizer_g.param_groups[0]["lr"],
                            lr_d=optimizer_d.param_groups[0]["lr"],
                            extras={
                                "Progress/nimg": float(cur_nimg),
                                "Progress/kimg": cur_kimg,
                                **metric_scalars,
                            },
                        )
                        print(f"中间评测 @ nimg={cur_nimg}, kimg={cur_kimg:.3f}: {eval_result['metrics']}")

                if should_eval and world_size > 1:
                    accelerator.wait_for_everyone()

                if reached_end:
                    break

            current_epoch += 1

        if is_rank0:
            print("DCGAN 训练完成。")
    except KeyboardInterrupt:
        if is_rank0:
            print("\n收到中断信号，正在清理训练进程并释放资源...")
    finally:
        accelerator.wait_for_everyone()
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
    """通过 accelerate 统一单卡/多卡训练入口。"""

    args = parse_args()
    args.output_dir = str(ensure_dir(args.output_dir))
    train(args)


if __name__ == "__main__":
    main()
