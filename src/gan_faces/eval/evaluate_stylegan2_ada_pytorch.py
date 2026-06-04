"""`stylegan2-ada-pytorch` 模型的 NVIDIA 指标评测入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

SRC_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STYLEGAN2_ADA_ROOT = PROJECT_ROOT.parent / "stylegan2-ada-pytorch"
sys.path.insert(0, str(SRC_ROOT))

from gan_faces.eval.nvidia_evaluator import evaluate_adapter, parse_metrics, supported_metrics
from gan_faces.utils import get_device, save_json, set_random_seed


def _add_stylegan2_ada_repo_to_path(stylegan2_ada_root: Path) -> None:
    if str(stylegan2_ada_root) not in sys.path:
        sys.path.insert(0, str(stylegan2_ada_root))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 NVIDIA StyleGAN2-ADA 指标评估官方 stylegan2-ada-pytorch 生成器")
    parser.add_argument("--checkpoint", type=str, required=True, help="官方仓库的 `network-snapshot-xxxxxx.pkl` 路径")
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="真实图像目录或 zip 路径，建议传入 CelebA 图片目录或 zip",
    )
    parser.add_argument(
        "--stylegan2-ada-root",
        type=str,
        default=str(DEFAULT_STYLEGAN2_ADA_ROOT),
        help="`stylegan2-ada-pytorch` 仓库根目录",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="fid5k",
        help="逗号分隔的指标名，例如 fid5k,kid5k,pr5k3,is5k；兼容 fid/is/both",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--verbose", action="store_true", help="打印 NVIDIA 指标计算进度")
    parser.add_argument("--no-cache", action="store_true", help="禁用真实特征缓存")
    parser.add_argument(
        "--output-json",
        type=str,
        default="outputs/metrics/stylegan2_ada_pytorch_nvidia_metrics.json",
        help="评测结果输出路径",
    )
    return parser.parse_args()


def build_stylegan2_ada_adapter(
    checkpoint_path: str | Path,
    stylegan2_ada_root: str | Path,
    device: torch.device,
):
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint_path}")

    stylegan2_ada_root = Path(stylegan2_ada_root).resolve()
    if not stylegan2_ada_root.exists():
        raise FileNotFoundError(f"`stylegan2-ada-pytorch` 仓库不存在: {stylegan2_ada_root}")

    _add_stylegan2_ada_repo_to_path(stylegan2_ada_root)
    import dnnlib
    import legacy

    with dnnlib.util.open_url(str(checkpoint_path), verbose=False) as f:
        network_dict = legacy.load_network_pkl(f)

    adapter = network_dict["G_ema"].eval().requires_grad_(False).to(device)
    metadata = {
        "training_set_kwargs": dict(network_dict.get("training_set_kwargs") or {}),
        "has_augment_pipe": network_dict.get("augment_pipe") is not None,
    }
    return adapter, metadata


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)
    device = get_device(args.device)
    metrics = parse_metrics(args.metrics)

    adapter, metadata = build_stylegan2_ada_adapter(
        checkpoint_path=args.checkpoint,
        stylegan2_ada_root=args.stylegan2_ada_root,
        device=device,
    )

    result = evaluate_adapter(
        adapter=adapter,
        data_path=args.data_root,
        metrics=metrics,
        verbose=args.verbose,
        cache=not args.no_cache,
    )
    result["checkpoint"] = str(Path(args.checkpoint).resolve())
    result["stylegan2_ada_root"] = str(Path(args.stylegan2_ada_root).resolve())
    result["model_type"] = "stylegan2-ada-pytorch"
    result["metric_backend"] = "nvidia_stylegan2_ada"
    result["requested_metrics"] = metrics
    result["device"] = str(device)
    result["checkpoint_metadata"] = metadata

    save_json(result, args.output_json)
    print("可用指标: " + ", ".join(supported_metrics()))
    print(f"评测完成，结果已保存到: {args.output_json}")
    for metric_name, metric_values in result["metrics"].items():
        print(f"{metric_name}: {metric_values}")


if __name__ == "__main__":
    main()

# python evaluate_stylegan2_ada_pytorch.py --checkpoint "..\stylegan2-ada-pytorch\training-runs\network-snapshot-009878.pkl" --stylegan2-ada-root "E:\exp\stylegan2-ada-pytorch" --data-root ".\data\celeba\img_align_celeba" --metrics fid5k --device cuda --verbose