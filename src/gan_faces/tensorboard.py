"""TensorBoard helpers for GAN training."""

import os
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import torch
from torchvision.utils import make_grid


def create_summary_writer(log_dir: Union[str, Path]) -> Any:
    """Create a TensorBoard writer, raising a clear error if the dependency is missing."""

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise ImportError(
            "TensorBoard is not installed. Run `pip install -r requirements.txt` "
            "or install `tensorboard` manually."
        ) from exc
    return SummaryWriter(log_dir=str(log_dir))


def add_training_scalars(
    writer: Optional[Any],
    global_step: int,
    loss_d: float,
    loss_g: float,
    d_real: float,
    d_fake: float,
    lr_g: float,
    lr_d: float,
    gp: Optional[float],
    wasserstein: Optional[float],
) -> None:
    """Log loss, discriminator confidence and learning rates."""

    if writer is None:
        return

    writer.add_scalar("Loss/loss_d", loss_d, global_step)
    writer.add_scalar("Loss/loss_g", loss_g, global_step)
    writer.add_scalar("Loss/d_real", d_real, global_step)
    writer.add_scalar("Loss/d_fake", d_fake, global_step)
    writer.add_scalar("LearningRate/generator", lr_g, global_step)
    writer.add_scalar("LearningRate/discriminator", lr_d, global_step)
    if gp is not None:
        writer.add_scalar("Loss/gradient_penalty", gp, global_step)
    if wasserstein is not None:
        writer.add_scalar("Loss/wasserstein", wasserstein, global_step)


def add_scalar_groups(writer: Optional[Any], scalars: Mapping[str, float], global_step: int) -> None:
    """批量写入按层级分组的标量。"""

    if writer is None:
        return
    for name, value in scalars.items():
        writer.add_scalar(name, value, global_step)


def add_text_block(writer: Optional[Any], tag: str, text: str, global_step: int = 0) -> None:
    """写入文本信息，方便记录实验配置。"""

    if writer is None:
        return
    writer.add_text(tag, text, global_step)


def add_sample_images(
    writer: Optional[Any],
    images: torch.Tensor,
    global_step: int,
    nrow: int = 8,
) -> None:
    """Log a normalized generated image grid."""

    if writer is None:
        return

    grid = make_grid(images.detach().cpu(), nrow=nrow, normalize=True, value_range=(-1, 1))
    writer.add_image("samples/fixed_noise", grid, global_step)


def close_summary_writer(writer: Optional[Any]) -> None:
    """Flush and close the TensorBoard writer if it was created."""

    if writer is not None:
        writer.flush()
        writer.close()
