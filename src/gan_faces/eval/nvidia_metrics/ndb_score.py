# Copyright (c) 2026.

"""Number of Statistically-Different Bins (NDB).

Adapted from GAN-Metrics/scores/ndb_jsd.py and wired into the local
StyleGAN2-ADA metric adapter.
"""

import copy
import math

import numpy as np
import torch
import torch.nn.functional as F

from .. import nvidia_dnnlib as dnnlib


#----------------------------------------------------------------------------

def _squared_distances(samples, centers, batch_size=1024):
    center_norms = np.sum(centers * centers, axis=1)
    distances = np.empty((samples.shape[0], centers.shape[0]), dtype=np.float32)

    for start in range(0, samples.shape[0], batch_size):
        end = min(start + batch_size, samples.shape[0])
        batch = samples[start:end]
        batch_norms = np.sum(batch * batch, axis=1, keepdims=True)
        distances[start:end] = batch_norms - 2.0 * batch @ centers.T + center_norms[None, :]

    return distances


def _assign_to_centers(samples, centers):
    return np.argmin(_squared_distances(samples, centers), axis=1)


def _kmeans(samples, number_of_bins, max_iter=100):
    if samples.shape[0] < number_of_bins:
        raise ValueError(
            f"NDB requires at least {number_of_bins} real samples; got {samples.shape[0]}"
        )

    init_indices = np.random.choice(samples.shape[0], number_of_bins, replace=False)
    centers = samples[init_indices].astype(np.float32, copy=True)
    labels = None

    for _iter in range(max_iter):
        new_labels = _assign_to_centers(samples, centers)
        if labels is not None and np.array_equal(new_labels, labels):
            break
        labels = new_labels

        for bin_idx in range(number_of_bins):
            members = samples[labels == bin_idx]
            if members.shape[0] == 0:
                centers[bin_idx] = samples[np.random.randint(samples.shape[0])]
            else:
                centers[bin_idx] = np.mean(members, axis=0)

    return _assign_to_centers(samples, centers), centers


def _kl_divergence(p, q):
    p_pos = p > 0
    return np.sum(p[p_pos] * np.log(p[p_pos] / q[p_pos]))


def _jensen_shannon_divergence(p, q):
    m = (p + q) * 0.5
    return 0.5 * (_kl_divergence(p, m) + _kl_divergence(q, m))


def _different_bins(p1, n1, p2, n2, significance_level=0.05, z_threshold=None):
    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = np.sqrt(p * (1.0 - p) * (1.0 / n1 + 1.0 / n2))

    if z_threshold is not None:
        return np.abs(p1 - p2) > z_threshold * se

    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.divide(p1 - p2, se, out=np.zeros_like(p1), where=se > 0)
    p_values = np.vectorize(math.erfc)(np.abs(z) / math.sqrt(2.0))
    return p_values < significance_level


class _NDB:
    def __init__(
        self,
        training_data,
        number_of_bins=100,
        significance_level=0.05,
        z_threshold=4.0,
        whitening=False,
        max_dims=None,
    ):
        self.number_of_bins = number_of_bins
        self.significance_level = significance_level
        self.z_threshold = z_threshold
        self.whitening = whitening
        self.ndb_eps = 1e-6
        self.training_mean = 0.0
        self.training_std = 1.0
        self.max_dims = max_dims
        self.bin_centers = None
        self.bin_proportions = None
        self.ref_sample_size = None
        self.used_d_indices = None

        self._construct_bins(training_data)

    def _construct_bins(self, training_samples):
        training_samples = np.asarray(training_samples, dtype=np.float32)
        n, d = training_samples.shape

        if self.whitening:
            self.training_mean = np.mean(training_samples, axis=0)
            self.training_std = np.std(training_samples, axis=0) + self.ndb_eps

        if self.max_dims is None and d > 1000:
            self.max_dims = d // 6

        whitened_samples = (training_samples - self.training_mean) / self.training_std
        d_used = d if self.max_dims is None else min(d, self.max_dims)
        self.used_d_indices = np.random.choice(d, d_used, replace=False)

        labels, used_centers = _kmeans(
            whitened_samples[:, self.used_d_indices],
            number_of_bins=self.number_of_bins,
        )
        bin_centers = np.zeros([self.number_of_bins, d], dtype=np.float32)
        label_counts = np.bincount(labels, minlength=self.number_of_bins)

        for bin_idx in range(self.number_of_bins):
            members = whitened_samples[labels == bin_idx]
            if members.shape[0] == 0:
                bin_centers[bin_idx, self.used_d_indices] = used_centers[bin_idx]
            else:
                bin_centers[bin_idx] = np.mean(members, axis=0)

        bin_order = np.argsort(-label_counts)
        self.bin_proportions = label_counts[bin_order] / np.sum(label_counts)
        self.bin_centers = bin_centers[bin_order]
        self.ref_sample_size = n

    def evaluate(self, query_samples):
        query_samples = np.asarray(query_samples, dtype=np.float32)
        query_bin_proportions = self._calculate_bin_proportions(query_samples)
        different_bins = _different_bins(
            self.bin_proportions,
            self.ref_sample_size,
            query_bin_proportions,
            query_samples.shape[0],
            significance_level=self.significance_level,
            z_threshold=self.z_threshold,
        )
        ndb = int(np.count_nonzero(different_bins))
        js = float(_jensen_shannon_divergence(self.bin_proportions, query_bin_proportions))
        return ndb, js

    def _calculate_bin_proportions(self, samples):
        assert self.bin_centers is not None
        assert samples.shape[1] == self.bin_centers.shape[1]

        whitened_samples = (samples - self.training_mean) / self.training_std
        labels = _assign_to_centers(
            whitened_samples[:, self.used_d_indices],
            self.bin_centers[:, self.used_d_indices],
        )
        label_counts = np.bincount(labels, minlength=self.number_of_bins)
        return label_counts / np.sum(label_counts)


#----------------------------------------------------------------------------

def _prepare_images(images, resolution, used_d_indices=None):
    if images.shape[1] == 1:
        images = images.repeat([1, 3, 1, 1])
    elif images.shape[1] > 3:
        images = images[:, :3]

    images = images.to(torch.float32)
    if images.shape[2] != resolution or images.shape[3] != resolution:
        images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)
    images = images.clamp(0, 255).div(255.0).flatten(1)
    if used_d_indices is not None:
        index = torch.as_tensor(used_d_indices, dtype=torch.long, device=images.device)
        images = images.index_select(1, index)
    return images.cpu().numpy().astype(np.float32)


def _collect_real_samples(opts, resolution, max_items, batch_size=64, used_d_indices=None):
    dataset_kwargs = dnnlib.EasyDict(opts.dataset_kwargs)
    dataset_kwargs.update(resolution=resolution, max_size=None, xflip=False)
    dataset = dnnlib.util.construct_class_by_name(**dataset_kwargs)

    if max_items is None:
        num_items = len(dataset)
    else:
        num_items = min(len(dataset), max_items)

    if opts.num_gpus != 1:
        raise ValueError("NDB evaluation currently supports num_gpus=1")

    if torch.cuda.is_available() and opts.device.type == "cuda":
        data_loader_kwargs = dict(pin_memory=True, num_workers=0)
    else:
        data_loader_kwargs = dict(pin_memory=False, num_workers=0)

    samples = []
    progress = opts.progress.sub(tag="real images", num_items=num_items, rel_lo=0, rel_hi=0.25)
    for images, _labels in torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=list(range(num_items)),
        batch_size=batch_size,
        **data_loader_kwargs,
    ):
        samples.append(_prepare_images(images, resolution, used_d_indices=used_d_indices))
        progress.update(sum(sample.shape[0] for sample in samples))

    return np.concatenate(samples, axis=0)


def _collect_generated_samples(opts, resolution, num_gen, batch_size=64, batch_gen=None, used_d_indices=None):
    if opts.num_gpus != 1:
        raise ValueError("NDB evaluation currently supports num_gpus=1")

    if batch_gen is None:
        batch_gen = min(batch_size, 4)
    assert batch_size % batch_gen == 0

    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)

    samples = []
    num_items = 0
    progress = opts.progress.sub(tag="generated images", num_items=num_gen, rel_lo=0.25, rel_hi=1)

    while num_items < num_gen:
        images = []
        for _i in range(batch_size // batch_gen):
            if num_items + sum(batch.shape[0] for batch in images) >= num_gen:
                break
            z = torch.randn([batch_gen, G.z_dim], device=opts.device)
            c = [dataset.get_label(np.random.randint(len(dataset))) for _i in range(batch_gen)]
            c = torch.from_numpy(np.stack(c))
            if opts.device.type == "cuda":
                c = c.pin_memory()
            c = c.to(opts.device)
            img = G(z=z, c=c, **opts.G_kwargs)
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            images.append(img)

        images = torch.cat(images, dim=0)[: num_gen - num_items]
        samples.append(_prepare_images(images, resolution, used_d_indices=used_d_indices))
        num_items += samples[-1].shape[0]
        progress.update(num_items)

    return np.concatenate(samples, axis=0)


def compute_ndb(
    opts,
    max_real,
    num_gen,
    number_of_bins=100,
    resolution=128,
    z_threshold=4.0,
    whitening=False,
    max_dims=None,
):
    num_dims = resolution * resolution * 3
    used_d_indices = None
    ndb_max_dims = max_dims
    if num_dims > 1000 or max_dims is not None:
        dims_to_use = num_dims // 6 if max_dims is None else min(num_dims, max_dims)
        if dims_to_use < num_dims:
            used_d_indices = np.random.choice(num_dims, dims_to_use, replace=False)
            ndb_max_dims = dims_to_use

    real_samples = _collect_real_samples(
        opts,
        resolution=resolution,
        max_items=max_real,
        used_d_indices=used_d_indices,
    )
    gen_samples = _collect_generated_samples(
        opts,
        resolution=resolution,
        num_gen=num_gen,
        used_d_indices=used_d_indices,
    )

    ndb = _NDB(
        training_data=real_samples,
        number_of_bins=number_of_bins,
        z_threshold=z_threshold,
        whitening=whitening,
        max_dims=ndb_max_dims,
    )
    ndb_value, js_value = ndb.evaluate(gen_samples)
    return ndb_value, ndb_value / number_of_bins, js_value
#----------------------------------------------------------------------------
