from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .checkpointing import mark_success, save_json, stage_is_complete
from .config import Config


def _file(config: Config, key: str) -> Path:
    value = config.data.get(key)
    if not value:
        raise ValueError(f"missing data.{key} in configuration")
    path = Path(value)
    if not path.is_absolute():
        path = (config.path.parent / path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _date_slice(values: np.ndarray, period: list[str] | tuple[str, str]) -> np.ndarray:
    dates = pd.to_datetime(values)
    return (dates >= pd.Timestamp(period[0])) & (dates <= pd.Timestamp(period[1]))


def _coord_name(ds: xr.Dataset, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in ds.coords:
            return name
    raise ValueError(f"could not find coordinate among {candidates}")


def _load_pair(config: Config) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    channels = config.data["input_channels"]
    targets = config.data["target_channels"]
    try:
        import xesmf as xe
    except ImportError as error:
        raise RuntimeError(
            "preprocessing requires xESMF bilinear regridding; install the data extra"
        ) from error

    with xr.open_dataset(_file(config, "gcm_file")) as gcm, xr.open_dataset(_file(config, "era5_file")) as era5:
        lat = _coord_name(era5, ("lat", "latitude"))
        lon = _coord_name(era5, ("lon", "longitude"))
        gcm_lat = _coord_name(gcm, ("lat", "latitude"))
        gcm_lon = _coord_name(gcm, ("lon", "longitude"))
        time = _coord_name(era5, ("time", "date"))
        gcm_time = _coord_name(gcm, ("time", "date"))
        if gcm.sizes[gcm_lat] != 36 or gcm.sizes[gcm_lon] != 27:
            raise ValueError(f"expected GCM core grid 36x27, got {gcm.sizes[gcm_lat]}x{gcm.sizes[gcm_lon]}")
        if era5.sizes[lat] != 181 or era5.sizes[lon] != 201:
            raise ValueError(f"expected ERA5 grid 181x201, got {era5.sizes[lat]}x{era5.sizes[lon]}")
        dates = pd.to_datetime(era5[time].values)
        gcm_dates = pd.to_datetime(gcm[gcm_time].values)
        common = np.intersect1d(
            dates.values.astype("datetime64[ns]"),
            gcm_dates.values.astype("datetime64[ns]"),
        )
        if len(common) == 0:
            raise ValueError("GCM and ERA5 have no common dates")

        regridder = xe.Regridder(gcm, era5, method="bilinear", periodic=False)
        x_values, y_values = [], []
        for channel in channels:
            source = channel.removesuffix("_gcm")
            if source not in gcm:
                raise ValueError(f"missing GCM variable {source}")
            mapped = regridder(gcm[source])
            mapped = mapped.sel({gcm_time: common}) if gcm_time in mapped.dims else mapped
            x_values.append(mapped.values.astype("float32"))
        for channel in targets:
            source = channel.removesuffix("_era5")
            if source not in era5:
                raise ValueError(f"missing ERA5 variable {source}")
            values = era5[source].sel({time: common}).values.astype("float32")
            y_values.append(values)
    x = np.stack(x_values, axis=1)
    y = np.stack(y_values, axis=1)
    if x.shape != y.shape:
        raise ValueError(f"input and target shapes differ after alignment: {x.shape} vs {y.shape}")
    return x, y, {
        "dates": [str(v)[:10] for v in common],
        "lat": era5[lat].values.tolist(),
        "lon": era5[lon].values.tolist(),
        "regridding": "xesmf_bilinear",
    }


def _period_indices(dates: list[str], periods: dict[str, Any]) -> dict[str, np.ndarray]:
    result = {}
    for name, period in periods.items():
        mask = _date_slice(np.asarray(dates, dtype="datetime64[ns]"), period)
        result[name] = np.flatnonzero(mask)
        if len(result[name]) == 0:
            raise ValueError(f"period {name} contains no samples")
    return result


def run_preprocess(config: Config, force: bool = False) -> Path:
    stage = config.output_root / "preprocessing"
    if stage_is_complete(stage) and not force:
        return stage
    stage.mkdir(parents=True, exist_ok=True)
    x, y, metadata = _load_pair(config)
    indices = _period_indices(metadata["dates"], config.periods)
    train = indices.get("train", indices.get("production"))
    if train is None:
        raise ValueError("periods must define train or production")
    pr_index = config.data["input_channels"].index("pr_gcm")
    target_pr_index = config.data["target_channels"].index("pr_era5")
    if pr_index != target_pr_index:
        raise ValueError("input and target precipitation channels must have the same index")
    precipitation_transform = config.raw.get("preprocessing", {}).get("precipitation_transform", "log1p")
    log1p_applied = precipitation_transform == "log1p"
    if log1p_applied:
        x[:, pr_index] = np.log1p(np.maximum(x[:, pr_index], 0))
        y[:, target_pr_index] = np.log1p(np.maximum(y[:, target_pr_index], 0))

    x_mean = np.nanmean(x[train], axis=(0, 2, 3), keepdims=True)
    y_mean = np.nanmean(y[train], axis=(0, 2, 3), keepdims=True)
    if not np.isfinite(x_mean).all() or not np.isfinite(y_mean).all():
        raise ValueError("a training channel contains only NaN or non-finite values")
    for channel in range(x.shape[1]):
        x[:, channel] = np.nan_to_num(x[:, channel], nan=float(x_mean.reshape(-1)[channel]))
        y[:, channel] = np.nan_to_num(y[:, channel], nan=float(y_mean.reshape(-1)[channel]))
    x_std = x[train].std(axis=(0, 2, 3), keepdims=True)
    y_std = y[train].std(axis=(0, 2, 3), keepdims=True)
    x_std = x_std + 1e-6
    y_std = y_std + 1e-6
    np.save(stage / "X.npy", ((x - x_mean) / x_std).astype("float32"))
    np.save(stage / "Y.npy", ((y - y_mean) / y_std).astype("float32"))
    np.savez(stage / "splits.npz", **indices)
    norm = {
        "normalization_version": 2,
        "X_mean": x_mean.reshape(-1).tolist(),
        "X_std": x_std.reshape(-1).tolist(),
        "Y_mean": y_mean.reshape(-1).tolist(),
        "Y_std": y_std.reshape(-1).tolist(),
        "input_channels": config.data["input_channels"],
        "target_channels": config.data["target_channels"],
        "pr_index": pr_index,
        "target_pr_index": target_pr_index,
        "precipitation_transform": precipitation_transform,
        "log1p_applied": log1p_applied,
        "log1p_input_applied": log1p_applied,
        "log1p_target_applied": log1p_applied,
        "split": "production" if "production" in config.periods else "dev",
        "edge_fill_values": x_mean.reshape(-1).tolist(),
        "target_edge_fill_values": y_mean.reshape(-1).tolist(),
        "nan_fill_policy": "training_split_channel_mean",
    }
    save_json(norm, stage / "normalization.json")
    metadata.update({"shape": list(x.shape), "config": config.raw, "normalization": norm})
    save_json(metadata, stage / "metadata.json")
    fingerprint = hashlib.sha256(json.dumps(metadata, sort_keys=True, default=str).encode()).hexdigest()
    mark_success(stage, {"fingerprint": fingerprint, "shape": list(x.shape)})
    return stage
