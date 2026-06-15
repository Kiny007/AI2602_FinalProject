# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import time
import torch
from .. import nvidia_dnnlib as dnnlib

from . import frechet_inception_distance
from . import inception_score
from . import kernel_inception_distance
from . import ndb_score
from . import precision_recall

#----------------------------------------------------------------------------

_metric_dict = dict() # name => fn

def register_metric(fn):
    assert callable(fn)
    _metric_dict[fn.__name__] = fn
    return fn

def is_valid_metric(metric):
    return metric in _metric_dict

def list_valid_metrics():
    return list(_metric_dict.keys())

#----------------------------------------------------------------------------

def calc_metric(metric, **kwargs):
    assert is_valid_metric(metric)
    from . import metric_utils
    opts = metric_utils.MetricOptions(**kwargs)

    # Calculate.
    start_time = time.time()
    results = _metric_dict[metric](opts)
    total_time = time.time() - start_time

    # Broadcast results.
    for key, value in list(results.items()):
        if opts.num_gpus > 1:
            value = torch.as_tensor(value, dtype=torch.float64, device=opts.device)
            torch.distributed.broadcast(tensor=value, src=0)
            value = float(value.cpu())
        results[key] = value

    # Decorate with metadata.
    return dnnlib.EasyDict(
        results         = dnnlib.EasyDict(results),
        metric          = metric,
        total_time      = total_time,
        total_time_str  = dnnlib.util.format_time(total_time),
        num_gpus        = opts.num_gpus,
    )

#----------------------------------------------------------------------------
# Primary metrics.

@register_metric
def fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    fid = frechet_inception_distance.compute_fid(opts, max_real=None, num_gen=50000)
    return dict(fid50k_full=fid)

@register_metric
def kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    kid = kernel_inception_distance.compute_kid(opts, max_real=1000000, num_gen=50000, num_subsets=100, max_subset_size=1000)
    return dict(kid50k_full=kid)

@register_metric
def pr50k3_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    precision, recall = precision_recall.compute_pr(opts, max_real=200000, num_gen=50000, nhood_size=3, row_batch_size=10000, col_batch_size=10000)
    return dict(pr50k3_full_precision=precision, pr50k3_full_recall=recall)

@register_metric
def is5k(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    mean, std = inception_score.compute_is(opts, num_gen=5000, num_splits=10)
    return dict(is5k_mean=mean, is5k_std=std)

#----------------------------------------------------------------------------
# Legacy metrics.

@register_metric
def fid5k(opts):
    opts.dataset_kwargs.update(max_size=None)
    fid = frechet_inception_distance.compute_fid(opts, max_real=5000, num_gen=5000)
    return dict(fid5k=fid)

@register_metric
def kid50k(opts):
    opts.dataset_kwargs.update(max_size=None)
    kid = kernel_inception_distance.compute_kid(opts, max_real=5000, num_gen=5000, num_subsets=100, max_subset_size=1000)
    return dict(kid5k=kid)

@register_metric
def pr5k3(opts):
    opts.dataset_kwargs.update(max_size=None)
    precision, recall = precision_recall.compute_pr(opts, max_real=5000, num_gen=5000, nhood_size=3, row_batch_size=10000, col_batch_size=10000)
    return dict(pr5k3_precision=precision, pr5k3_recall=recall)

@register_metric
def ndb5k(opts):
    opts.dataset_kwargs.update(max_size=None)
    ndb, ndb_ratio, js = ndb_score.compute_ndb(opts, max_real=5000, num_gen=5000, number_of_bins=100)
    return dict(ndb5k=ndb, ndb5k_ratio=ndb_ratio, ndb5k_jsd=js)

#----------------------------------------------------------------------------
