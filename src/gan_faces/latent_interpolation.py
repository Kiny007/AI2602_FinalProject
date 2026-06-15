"""Latent-space interpolation helpers."""

from __future__ import annotations

import torch


def _broadcast_alpha(alpha: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    while alpha.ndim < reference.ndim:
        alpha = alpha.unsqueeze(-1)
    return alpha


def linear_interpolate(start: torch.Tensor, end: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Linear interpolation: (1 - alpha) * start + alpha * end."""

    return torch.lerp(start, end, _broadcast_alpha(alpha, start))


def normalize_latent(x: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    """Normalize latent vectors along the latent dimension."""

    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def spherical_interpolate(
    start: torch.Tensor,
    end: torch.Tensor,
    alpha: torch.Tensor,
    dim: int = 1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Spherical interpolation for Gaussian latent spaces.

    This keeps the path on the hypersphere defined by the two endpoint
    directions, which is usually a better fit for normally sampled z vectors.
    """

    alpha = _broadcast_alpha(alpha, start)
    start_norm = normalize_latent(start, dim=dim, eps=eps)
    end_norm = normalize_latent(end, dim=dim, eps=eps)
    dot = (start_norm * end_norm).sum(dim=dim, keepdim=True).clamp(-1 + eps, 1 - eps)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)

    coef_start = torch.sin((1 - alpha) * omega) / sin_omega.clamp_min(eps)
    coef_end = torch.sin(alpha * omega) / sin_omega.clamp_min(eps)
    spherical = coef_start * start + coef_end * end
    linear = linear_interpolate(start, end, alpha)
    return torch.where(sin_omega.abs() < eps, linear, spherical)
