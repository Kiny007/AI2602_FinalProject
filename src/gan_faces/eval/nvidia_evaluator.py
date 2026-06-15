"""NVIDIA 评测标准接入入口。"""

from __future__ import annotations

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


def evaluate_adapter(
    adapter: torch.nn.Module,
    data_path: str | Path,
    metrics: list[str],
    verbose: bool = True,
    cache: bool = True,
    num_gpus: int = 1,
    rank: int = 0,
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
            num_gpus=num_gpus,
            rank=rank,
            device=next(adapter.parameters()).device,
            progress=progress,
            cache=cache,
        )
        metric_results[metric] = dict(result.results)

    return {
        "data_path": str(data_path),
        "image_resolution": adapter.img_resolution,
        "metrics": metric_results,
    }
