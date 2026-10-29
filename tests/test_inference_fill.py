"""Regression test for the regrid border fill used at inference time."""

from __future__ import annotations

import concurrent.futures

import numpy as np

from truss_downscaling.inference import _collect_regrid_chunk
from truss_downscaling.transforms import standardize_inputs

PR_MEAN = 1.29
PR_STD = 1.55


def test_inference_border_fill_reproduces_the_training_value():
    """Filling before log1p must use the physical inverse of the stored log-space
    edge value, otherwise the border is shifted relative to training."""
    x_mean = np.array([25.0, 70.0, PR_MEAN], dtype=np.float32)
    x_std = np.array([5.0, 10.0, PR_STD], dtype=np.float32)
    norm = {
        "normalization_version": 2,
        "X_mean": x_mean.tolist(),
        "X_std": x_std.tolist(),
        "Y_mean": x_mean.tolist(),
        "Y_std": x_std.tolist(),
        "pr_index": 2,
        "target_pr_index": 2,
        "log1p_input_applied": True,
        "log1p_target_applied": True,
    }

    futures = []
    for _ in range(3):
        future: concurrent.futures.Future = concurrent.futures.Future()
        future.set_result(np.full((1, 4, 4), np.nan, dtype=np.float32))
        futures.append(future)
    fill_values = x_mean.copy()
    fill_values[2] = np.expm1(max(float(fill_values[2]), 0.0))
    collected = _collect_regrid_chunk(futures, slice(0, 1), 3, 4, 4, fill_values)
    standardized = standardize_inputs(collected, norm)

    # Training filled the border *after* log1p with the log-space mean, so the
    # standardized border must be exactly zero for every channel, including pr.
    assert np.allclose(standardized, 0.0, atol=1e-5), standardized
