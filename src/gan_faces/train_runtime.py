"""训练运行时辅助工具。

本模块抽取出训练脚本共用的工程能力：多进程/多卡初始化、跨进程标量聚合、
EMA 权重维护、训练日志写入，以及样例图与 checkpoint 的统一保存。
"""

from __future__ import annotations

import csv
import gc
import json
import os
import random
import signal
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from .tensorboard import (
    add_sample_images,
    add_scalar_groups,
    add_text_block,
    add_training_scalars,
    close_summary_writer,
    create_summary_writer,
)
from .utils import ensure_dir, save_generated_grid


_INTERRUPT_REQUESTED = False
_REGISTERED_SIGNALS: dict[int, Any] = {}


@dataclass(frozen=True)
class TrainingLayout:
    """统一描述一次训练运行的输出目录布局。"""

    output_dir: Path
    sample_dir: Path
    checkpoint_dir: Path
    tensorboard_dir: Path
    csv_log_path: Path
    jsonl_log_path: Path
    options_path: Path


def find_free_port() -> str:
    """在本机挑一个可用端口，供 DDP 初始化时使用。"""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def seed_everything(seed: int) -> None:
    """固定 Python、NumPy 和 PyTorch 随机种子。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_device(device_arg: str, rank: int, world_size: int) -> torch.device:
    """根据命令行参数和 rank 选择当前进程使用的设备。"""

    if world_size > 1 and device_arg not in {"auto", "cuda"}:
        raise ValueError("多卡训练仅支持 `--device auto` 或 `--device cuda`")

    if device_arg == "auto":
        if torch.cuda.is_available():
            if world_size > torch.cuda.device_count():
                raise ValueError(f"请求了 {world_size} 张 GPU，但当前仅检测到 {torch.cuda.device_count()} 张")
            return torch.device("cuda", rank if world_size > 1 else 0)
        if world_size > 1:
            raise ValueError("未检测到 CUDA 设备，无法启动多卡训练")
        return torch.device("cpu")

    device = torch.device(device_arg)
    if world_size > 1 and device.type != "cuda":
        raise ValueError("多卡训练仅支持 CUDA 设备")
    if device.type == "cuda" and world_size > 1:
        return torch.device("cuda", rank)
    return device


def get_distributed_backend(device: torch.device) -> str:
    """在当前平台上为 DDP 选择合适的通信后端。"""

    if device.type == "cuda" and os.name != "nt" and dist.is_nccl_available():
        return "nccl"
    return "gloo"


def setup_distributed(rank: int, world_size: int, device: torch.device, master_addr: str, master_port: str) -> None:
    """初始化分布式进程组；单卡时直接跳过。"""

    cuda_index = 0 if device.index is None else device.index
    if world_size <= 1:
        if device.type == "cuda":
            torch.cuda.set_device(cuda_index)
        return

    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", master_port)
    if device.type == "cuda":
        torch.cuda.set_device(cuda_index)

    dist.init_process_group(
        backend=get_distributed_backend(device),
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size,
    )


def install_signal_handlers() -> None:
    """注册轻量 signal handler，在收到中断时请求训练循环尽快收尾。"""

    global _INTERRUPT_REQUESTED
    _INTERRUPT_REQUESTED = False

    def _handle_interrupt(signum, _frame) -> None:
        global _INTERRUPT_REQUESTED
        _INTERRUPT_REQUESTED = True

    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is None or sig in _REGISTERED_SIGNALS:
            continue
        _REGISTERED_SIGNALS[sig] = signal.getsignal(sig)
        signal.signal(sig, _handle_interrupt)


def restore_signal_handlers() -> None:
    """恢复此前的 signal handler，避免影响后续进程逻辑。"""

    global _INTERRUPT_REQUESTED
    for sig, handler in list(_REGISTERED_SIGNALS.items()):
        signal.signal(sig, handler)
        _REGISTERED_SIGNALS.pop(sig, None)
    _INTERRUPT_REQUESTED = False


def interrupt_requested() -> bool:
    """返回当前进程是否收到了终止训练的信号。"""

    return _INTERRUPT_REQUESTED


def cleanup_distributed() -> None:
    """销毁分布式进程组。"""

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    """在多卡训练时等待所有进程同步。"""

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def shutdown_dataloader_workers(dataloader: Any | None) -> None:
    """显式关闭 DataLoader worker，降低 Ctrl+C 后残留子进程的概率。"""

    if dataloader is None:
        return
    iterator = getattr(dataloader, "_iterator", None)
    shutdown_fn = getattr(iterator, "_shutdown_workers", None)
    if shutdown_fn is not None:
        shutdown_fn()


def release_cuda_memory(device: torch.device | None) -> None:
    """尽量释放当前进程持有的 CUDA 缓存。"""

    if device is None or device.type != "cuda" or not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize(device)
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    ipc_collect = getattr(torch.cuda, "ipc_collect", None)
    if ipc_collect is not None:
        try:
            ipc_collect()
        except Exception:
            pass


def cleanup_training_process(
    *,
    logger: "TrainingLogger | None" = None,
    dataloader: Any | None = None,
    device: torch.device | None = None,
    modules: list[Any] | None = None,
) -> None:
    """统一执行训练进程收尾，尽量释放 worker、DDP 和 CUDA 相关资源。"""

    shutdown_dataloader_workers(dataloader)
    if logger is not None:
        try:
            logger.close()
        except Exception:
            pass
    if modules:
        modules.clear()
    cleanup_distributed()
    gc.collect()
    release_cuda_memory(device)


def unwrap_module(module: nn.Module) -> nn.Module:
    """从 DDP 包装中取回原始模块。"""

    if isinstance(module, DistributedDataParallel):
        return module.module
    return module


def maybe_wrap_ddp(module: nn.Module, device: torch.device, world_size: int) -> nn.Module:
    """按需用 DDP 包装模型；单卡时保持原样。"""

    if world_size <= 1:
        return module

    kwargs: dict[str, Any] = {"broadcast_buffers": False}
    if device.type == "cuda":
        kwargs["device_ids"] = [device.index]
    return DistributedDataParallel(module, **kwargs)


def average_tensor(value: torch.Tensor, world_size: int) -> torch.Tensor:
    """把各进程上的标量张量取平均，用于统一日志输出。"""

    result = value.detach().clone()
    if world_size > 1:
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        result /= world_size
    return result


@torch.no_grad()
def update_ema(ema_model: nn.Module | None, source_model: nn.Module, decay: float) -> None:
    """更新生成器 EMA 权重；仅 rank0 会维护这一份副本。"""

    if ema_model is None:
        return

    source = unwrap_module(source_model)
    for ema_param, src_param in zip(ema_model.parameters(), source.parameters()):
        ema_param.mul_(decay).add_(src_param.detach(), alpha=1.0 - decay)
    for ema_buffer, src_buffer in zip(ema_model.buffers(), source.buffers()):
        ema_buffer.copy_(src_buffer.detach())


def create_training_layout(output_dir: str | Path, tensorboard_dir: str | Path = "") -> TrainingLayout:
    """创建并返回统一的训练输出目录结构。"""

    output_dir = ensure_dir(output_dir)
    sample_dir = ensure_dir(output_dir / "samples")
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    tensorboard_dir = ensure_dir(tensorboard_dir or output_dir / "tensorboard")
    return TrainingLayout(
        output_dir=output_dir,
        sample_dir=sample_dir,
        checkpoint_dir=checkpoint_dir,
        tensorboard_dir=tensorboard_dir,
        csv_log_path=output_dir / "train_log.csv",
        jsonl_log_path=output_dir / "stats.jsonl",
        options_path=output_dir / "training_options.json",
    )


class TrainingLogger:
    """统一管理控制台、CSV、JSONL 与 TensorBoard 日志。"""

    def __init__(self, layout: TrainingLayout, enable_tensorboard: bool) -> None:
        self.layout = layout
        self.writer = create_summary_writer(layout.tensorboard_dir) if enable_tensorboard else None
        self._write_csv_header_if_needed()
        self.jsonl_file = layout.jsonl_log_path.open("a", encoding="utf-8")

    def log_config(self, options: dict[str, Any]) -> None:
        """把运行配置写入 TensorBoard 文本面板。"""

        add_text_block(
            self.writer,
            "run/config",
            "```json\n" + json.dumps(options, indent=2, ensure_ascii=False) + "\n```",
            global_step=0,
        )

    def _write_csv_header_if_needed(self) -> None:
        if self.layout.csv_log_path.exists():
            return
        with self.layout.csv_log_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "epoch",
                    "step",
                    "nimg",
                    "kimg",
                    "loss_d",
                    "loss_g",
                    "d_real",
                    "d_fake",
                    "lr_g",
                    "lr_d",
                ]
            )

    def log_step(
        self,
        *,
        epoch: int,
        step: int,
        total_steps: int,
        global_step: int,
        metrics: dict[str, float],
        lr_g: float,
        lr_d: float,
        extras: dict[str, float] | None = None,
    ) -> None:
        """写入一步训练统计，同时同步到 TensorBoard 和 JSONL。"""

        row = [
            epoch,
            step,
            extras.get("Progress/nimg", "") if extras else "",
            extras.get("Progress/kimg", "") if extras else "",
            metrics["loss_d"],
            metrics["loss_g"],
            metrics["d_real"],
            metrics["d_fake"],
            lr_g,
            lr_d,
        ]
        with self.layout.csv_log_path.open("a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(row)

        record = {
            "timestamp": time.time(),
            "epoch": epoch,
            "step": step,
            "total_steps": total_steps,
            "global_step": global_step,
            **metrics,
            "lr_g": lr_g,
            "lr_d": lr_d,
        }
        if extras:
            record.update(extras)
        self.jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.jsonl_file.flush()

        add_training_scalars(
            writer=self.writer,
            global_step=global_step,
            loss_d=metrics["loss_d"],
            loss_g=metrics["loss_g"],
            d_real=metrics["d_real"],
            d_fake=metrics["d_fake"],
            lr_g=lr_g,
            lr_d=lr_d,
        )
        if extras:
            add_scalar_groups(self.writer, extras, global_step)

    def log_epoch(
        self,
        *,
        epoch: int,
        global_step: int,
        metrics: dict[str, float],
        elapsed_sec: float,
        extras: dict[str, float] | None = None,
    ) -> None:
        """记录 epoch 级别统计。"""

        record = {
            "timestamp": time.time(),
            "epoch": epoch,
            "global_step": global_step,
            "elapsed_sec": elapsed_sec,
            "scope": "epoch",
            **metrics,
        }
        if extras:
            record.update(extras)
        self.jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.jsonl_file.flush()

        if self.writer is not None:
            for name, value in metrics.items():
                self.writer.add_scalar(f"epoch/{name}", value, epoch)
            self.writer.add_scalar("epoch/elapsed_sec", elapsed_sec, epoch)
            add_scalar_groups(
                self.writer,
                {
                    "Progress/epoch": float(epoch),
                    "Timing/epoch_sec": elapsed_sec,
                    **(extras or {}),
                },
                global_step,
            )

    def save_samples(self, images: torch.Tensor, output_path: str | Path, global_step: int, nrow: int = 8) -> None:
        """保存样例图并同步到 TensorBoard。"""

        save_generated_grid(images, output_path, nrow=nrow)
        add_sample_images(self.writer, images, global_step, nrow=nrow)

    def close(self) -> None:
        """关闭所有日志句柄。"""

        self.jsonl_file.flush()
        self.jsonl_file.close()
        close_summary_writer(self.writer)


def save_training_options(layout: TrainingLayout, options: dict[str, Any]) -> None:
    """保存一次训练运行的配置，方便复现实验。"""

    layout.options_path.write_text(json.dumps(options, indent=2, ensure_ascii=False), encoding="utf-8")
