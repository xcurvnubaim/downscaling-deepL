from __future__ import annotations

from typing import Any

import numpy as np


NORMALIZATION_VERSION = 2


def validate_norm_stats(norm: dict[str, Any]) -> None:
    if int(norm.get("normalization_version", 0)) != NORMALIZATION_VERSION:
        raise RuntimeError(
            "normalization artifacts are from an older pipeline version; "
            "rerun preprocessing before training, evaluation, or inference"
        )
    required = (
        "X_mean",
        "X_std",
        "Y_mean",
        "Y_std",
        "pr_index",
        "target_pr_index",
        "log1p_input_applied",
        "log1p_target_applied",
    )
    missing = [key for key in required if key not in norm]
    if missing:
        raise RuntimeError(f"normalization metadata is missing required fields: {', '.join(missing)}")


def _stats(norm: dict[str, Any], prefix: str) -> tuple[np.ndarray, np.ndarray]:
    means = np.asarray(norm[f"{prefix}_mean"], dtype=np.float32).reshape(1, -1, 1, 1)
    stds = np.asarray(norm[f"{prefix}_std"], dtype=np.float32).reshape(1, -1, 1, 1)
    return means, stds


def standardize_inputs(values: np.ndarray, norm: dict[str, Any]) -> np.ndarray:
    validate_norm_stats(norm)
    transformed = np.asarray(values, dtype=np.float32).copy()
    if norm["log1p_input_applied"]:
        index = int(norm["pr_index"])
        transformed[:, index] = np.log1p(np.maximum(transformed[:, index], 0))
    means, stds = _stats(norm, "X")
    return (transformed - means) / stds


def unstandardize_targets(values: np.ndarray, norm: dict[str, Any]) -> np.ndarray:
    validate_norm_stats(norm)
    means, stds = _stats(norm, "Y")
    return np.asarray(values, dtype=np.float32) * stds + means


def unstandardize_inputs(values: np.ndarray, norm: dict[str, Any]) -> np.ndarray:
    validate_norm_stats(norm)
    means, stds = _stats(norm, "X")
    return np.asarray(values, dtype=np.float32) * stds + means


def to_physical_targets(values: np.ndarray, norm: dict[str, Any]) -> np.ndarray:
    physical = unstandardize_targets(values, norm)
    if norm["log1p_target_applied"]:
        index = int(norm["target_pr_index"])
        physical[:, index] = np.expm1(np.maximum(physical[:, index], 0))
    return physical


def to_physical_inputs(values: np.ndarray, norm: dict[str, Any]) -> np.ndarray:
    physical = unstandardize_inputs(values, norm)
    if norm["log1p_input_applied"]:
        index = int(norm["pr_index"])
        physical[:, index] = np.expm1(np.maximum(physical[:, index], 0))
    return physical
