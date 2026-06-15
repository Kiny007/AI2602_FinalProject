"""NVIDIA 评测标准接入入口。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from . import nvidia_dnnlib as dnnlib
from .nvidia_metrics import metric_main, metric_utils


_METRIC_ALIASES = {
    "fid": "fid5k",
    "is": "is5k",
    "ndb": "ndb5k",
    "both": "fid5k,is5k",
    "ppl": "ppl_z,ppl_w",
}


def supported_metrics() -> list[str]:
    return metric_main.list_valid_metrics()


def parse_metrics(metric_text: str) -> list[str]:
    normalized = _METRIC_ALIASES.get(metric_text.strip().lower(), metric_text)
    metrics = [
        _METRIC_ALIASES.get(item.strip().lower(), item.strip())
        for item in normalized.split(",")
        if item.strip()
    ]
    invalid = [metric for metric in metrics if not metric_main.is_valid_metric(metric)]
    if invalid:
        raise ValueError(
            "不支持的评测指标: " + ", ".join(invalid) + "；可选值为: " + ", ".join(supported_metrics())
        )
    return metrics


def has_ppl_metric(metrics: list[str]) -> bool:
    return any(metric in {"ppl_z", "ppl_w"} for metric in metrics)


def add_ppl_metric_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ppl-num-samples",
        type=int,
        default=10,
        help="PPL 短跑指标的采样数量；默认沿用 PPL.ipynb 的 num_samples=10",
    )
    parser.add_argument(
        "--ppl-epsilon",
        type=float,
        default=1e-4,
        help="PPL 短跑指标的 epsilon；默认沿用 PPL.ipynb 的 eps=1e-4",
    )
    parser.add_argument(
        "--ppl-batch-size",
        type=int,
        default=2,
        help="PPL 计算时每批生成的潜变量对数量",
    )
    parser.add_argument(
        "--ppl-sampling",
        type=str,
        choices=["full", "end"],
        default="full",
        help="PPL 短跑指标的路径采样方式；full 表示 t 在 [0, 1] 上随机采样",
    )
    parser.add_argument(
        "--ppl-crop",
        action="store_true",
        help="计算 PPL 前使用 StyleGAN 官方中心裁剪；PPL.ipynb 默认不裁剪",
    )


def collect_ppl_metric_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "ppl_num_samples": args.ppl_num_samples,
        "ppl_epsilon": args.ppl_epsilon,
        "ppl_batch_size": args.ppl_batch_size,
        "ppl_sampling": args.ppl_sampling,
        "ppl_crop": args.ppl_crop,
    }


def evaluate_adapter(
    adapter: torch.nn.Module,
    data_path: str | Path,
    metrics: list[str],
    verbose: bool = True,
    cache: bool = True,
    metric_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_path}")

    dataset_kwargs = dnnlib.EasyDict(
        class_name="gan_faces.eval.nvidia_dataset.ImageFolderDataset",
        path=str(data_path),
        resolution=adapter.img_resolution,
        use_labels=False,
        xflip=False,
        max_size=None,
    )

    progress = metric_utils.ProgressMonitor(verbose=verbose)
    metric_results: dict[str, Any] = {}
    for metric in metrics:
        result = metric_main.calc_metric(
            metric=metric,
            G=adapter,
            dataset_kwargs=dataset_kwargs,
            num_gpus=1,
            rank=0,
            device=next(adapter.parameters()).device,
            progress=progress,
            cache=cache,
            metric_kwargs=metric_kwargs or {},
        )
        metric_results[metric] = dict(result.results)

    return {
        "data_path": str(data_path),
        "image_resolution": adapter.img_resolution,
        "metrics": metric_results,
    }
