# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Perceptual Path Length (PPL) metric.

This follows the PPL notebook formula by measuring LPIPS / epsilon**2 over
nearby interpolated latent points. StyleGAN-style generators can be evaluated
in either z space or w space.
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from ...latent_interpolation import linear_interpolate, spherical_interpolate
from .. import nvidia_dnnlib as dnnlib
from . import metric_utils


def _has_w_space(G: torch.nn.Module) -> bool:
    supports_w_space = getattr(G, "supports_w_space", None)
    if supports_w_space is not None:
        return bool(supports_w_space)
    return callable(getattr(G, "mapping", None)) and callable(getattr(G, "synthesis", None))


def _mapping(G: torch.nn.Module, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    try:
        return G.mapping(z=z, c=c)
    except TypeError:
        try:
            return G.mapping(z, c)
        except TypeError:
            return G.mapping(z)


def _synthesis(G: torch.nn.Module, ws: torch.Tensor, G_kwargs: dnnlib.EasyDict) -> torch.Tensor:
    try:
        return G.synthesis(ws=ws, noise_mode="const", force_fp32=True, **G_kwargs)
    except TypeError:
        try:
            return G.synthesis(ws=ws, **G_kwargs)
        except TypeError:
            try:
                return G.synthesis(ws, **G_kwargs)
            except TypeError:
                return G.synthesis(ws)


def _forward(G: torch.nn.Module, z: torch.Tensor, c: torch.Tensor, G_kwargs: dnnlib.EasyDict) -> torch.Tensor:
    try:
        return G(z=z, c=c, **G_kwargs)
    except TypeError:
        try:
            return G(z, c, **G_kwargs)
        except TypeError:
            return G(z)


def _randomize_noise_buffers(G: torch.nn.Module) -> None:
    for name, buf in G.named_buffers():
        if name.endswith(".noise_const"):
            buf.copy_(torch.randn_like(buf))


class PPLSampler(torch.nn.Module):
    def __init__(
        self,
        G: torch.nn.Module,
        G_kwargs: dnnlib.EasyDict,
        epsilon: float,
        space: str,
        sampling: str,
        crop: bool,
        vgg16: torch.nn.Module,
    ) -> None:
        assert space in ["z", "w"]
        assert sampling in ["full", "end"]
        super().__init__()
        self.G = copy.deepcopy(G)
        self.G_kwargs = G_kwargs
        self.epsilon = epsilon
        self.space = space
        self.sampling = sampling
        self.crop = crop
        self.vgg16 = copy.deepcopy(vgg16)

    def _generate_from_w(self, w0: torch.Tensor, w1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        wt0 = linear_interpolate(w0, w1, t)
        wt1 = linear_interpolate(w0, w1, t + self.epsilon)

        _randomize_noise_buffers(self.G)
        return _synthesis(self.G, torch.cat([wt0, wt1]), self.G_kwargs)

    def _generate_from_z(self, z0: torch.Tensor, z1: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        zt0 = spherical_interpolate(z0, z1, t)
        zt1 = spherical_interpolate(z0, z1, t + self.epsilon)

        if _has_w_space(self.G):
            wt0, wt1 = _mapping(self.G, torch.cat([zt0, zt1]), torch.cat([c, c])).chunk(2)
            _randomize_noise_buffers(self.G)
            return _synthesis(self.G, torch.cat([wt0, wt1]), self.G_kwargs)

        img0 = _forward(self.G, zt0, c, self.G_kwargs)
        img1 = _forward(self.G, zt1, c, self.G_kwargs)
        return torch.cat([img0, img1])

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        t = torch.rand([c.shape[0], 1], device=c.device) * (1 if self.sampling == "full" else 0)
        z0, z1 = torch.randn([c.shape[0] * 2, self.G.z_dim], device=c.device).chunk(2)

        if self.space == "w":
            if not _has_w_space(self.G):
                raise ValueError("PPL in w-space requires a generator with mapping() and synthesis() methods.")
            w0, w1 = _mapping(self.G, torch.cat([z0, z1]), torch.cat([c, c])).chunk(2)
            img = self._generate_from_w(w0, w1, t)
        else:
            img = self._generate_from_z(z0, z1, t, c)

        if self.crop:
            assert img.shape[2] == img.shape[3]
            crop = img.shape[2] // 8
            img = img[:, :, crop * 3 : crop * 7, crop * 2 : crop * 6]

        factor = img.shape[2] // 256
        if factor > 1:
            img = img.reshape(
                [-1, img.shape[1], img.shape[2] // factor, factor, img.shape[3] // factor, factor]
            ).mean([3, 5])

        img = (img + 1) * (255 / 2)
        if img.shape[1] == 1:
            img = img.repeat([1, 3, 1, 1])

        lpips_t0, lpips_t1 = self.vgg16(img, resize_images=False, return_lpips=True).chunk(2)
        return (lpips_t0 - lpips_t1).square().sum(1) / self.epsilon**2


def compute_ppl(
    opts,
    num_samples: int,
    epsilon: float,
    space: str,
    sampling: str,
    crop: bool,
    batch_size: int,
    jit: bool = False,
) -> float:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive for PPL.")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive for PPL.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive for PPL.")

    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    vgg16_url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt"
    vgg16 = metric_utils.get_feature_detector(
        vgg16_url,
        device=opts.device,
        num_gpus=opts.num_gpus,
        rank=opts.rank,
        verbose=opts.progress.verbose,
    )

    sampler = PPLSampler(
        G=opts.G,
        G_kwargs=opts.G_kwargs,
        epsilon=epsilon,
        space=space,
        sampling=sampling,
        crop=crop,
        vgg16=vgg16,
    )
    sampler.eval().requires_grad_(False).to(opts.device)
    if jit:
        c = torch.zeros([batch_size, opts.G.c_dim], device=opts.device)
        sampler = torch.jit.trace(sampler, [c], check_trace=False)

    dist = []
    progress = opts.progress.sub(tag=f"ppl {space}-space", num_items=num_samples)
    for batch_start in range(0, num_samples, batch_size * opts.num_gpus):
        progress.update(batch_start)
        local_batch = min(batch_size, num_samples - batch_start)
        c = [dataset.get_label(np.random.randint(len(dataset))) for _i in range(local_batch)]
        c = torch.from_numpy(np.stack(c)).pin_memory().to(opts.device)
        x = sampler(c)
        for src in range(opts.num_gpus):
            y = x.clone()
            if opts.num_gpus > 1:
                torch.distributed.broadcast(y, src=src)
            dist.append(y)
    progress.update(num_samples)

    if opts.rank != 0:
        return float("nan")

    dist_array = torch.cat(dist)[:num_samples].cpu().numpy()
    lo = np.percentile(dist_array, 1, method="lower")
    hi = np.percentile(dist_array, 99, method="higher")
    filtered = dist_array[np.logical_and(dist_array >= lo, dist_array <= hi)]
    if filtered.size == 0:
        filtered = dist_array
    return float(filtered.mean())
