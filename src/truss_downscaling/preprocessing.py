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
    with xr.open_dataset(_file(config, "gcm_file")) as gcm, xr.open_dataset(_file(config, "era5_file")) as era5:
        lat = _coord_name(era5, ("lat", "latitude"))
        lon = _coord_name(era5, ("lon", "longitude"))
        gcm_lat = _coord_name(gcm, ("lat", "latitude"))
        gcm_lon = _coord_name(gcm, ("lon", "longitude"))
        time = _coord_name(era5, ("time", "date"))
        if gcm.sizes[gcm_lat] != 36 or gcm.sizes[gcm_lon] != 27:
            raise ValueError(f"expected GCM core grid 36x27, got {gcm.sizes[gcm_lat]}x{gcm.sizes[gcm_lon]}")
        if era5.sizes[lat] != 181 or era5.sizes[lon] != 201:
            raise ValueError(f"expected ERA5 grid 181x201, got {era5.sizes[lat]}x{era5.sizes[lon]}")
        dates = pd.to_datetime(era5[time].values)
        gcm_dates = pd.to_datetime(gcm[_coord_name(gcm, ("time", "date"))].values)
        common = np.intersect1d(dates.values.astype("datetime64[ns]"), gcm_dates.values.astype("datetime64[ns]"))
        if len(common) == 0:
            raise ValueError("GCM and ERA5 have no common dates")
        target_grid = {gcm_lat: era5[lat].values, gcm_lon: era5[lon].values}
        x_values, y_values = [], []
        for channel in channels:
            source = channel.removesuffix("_gcm")
            if source not in gcm:
                raise ValueError(f"missing GCM variable {source}")
            mapped = gcm[source].interp({gcm_lat: target_grid[gcm_lat], gcm_lon: target_grid[gcm_lon]}, method="linear")
            mapped = mapped.sel({"time": common}) if "time" in mapped.dims else mapped
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
    return x, y, {"dates": [str(v)[:10] for v in common], "lat": era5[lat].values.tolist(), "lon": era5[lon].values.tolist()}


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
    if config.raw.get("preprocessing", {}).get("precipitation_transform", "log1p") == "log1p":
        x[:, pr_index] = np.log1p(np.maximum(x[:, pr_index], 0))
    x_mean = np.nanmean(x[train], axis=(0, 2, 3), keepdims=True)
    y_mean = np.nanmean(y[train], axis=(0, 2, 3), keepdims=True)
    for channel in range(x.shape[1]):
        x[:, channel] = np.nan_to_num(x[:, channel], nan=float(x_mean.reshape(-1)[channel]))
        y[:, channel] = np.nan_to_num(y[:, channel], nan=float(y_mean.reshape(-1)[channel]))
    x_std = x[train].std(axis=(0, 2, 3), keepdims=True)
    y_std = y[train].std(axis=(0, 2, 3), keepdims=True)
    x_std[x_std < 1e-6] = 1
    y_std[y_std < 1e-6] = 1
    np.save(stage / "X.npy", ((x - x_mean) / x_std).astype("float32"))
    np.save(stage / "Y.npy", ((y - y_mean) / y_std).astype("float32"))
    np.savez(stage / "splits.npz", **indices)
    norm = {"X_mean": x_mean.reshape(-1).tolist(), "X_std": x_std.reshape(-1).tolist(), "Y_mean": y_mean.reshape(-1).tolist(), "Y_std": y_std.reshape(-1).tolist(), "input_channels": config.data["input_channels"], "target_channels": config.data["target_channels"], "pr_index": pr_index, "log1p_applied": True, "split": "production" if "production" in config.periods else "dev", "edge_fill_values": x_mean.reshape(-1).tolist()}
    save_json(norm, stage / "normalization.json")
    metadata.update({"shape": list(x.shape), "config": config.raw, "normalization": norm})
    save_json(metadata, stage / "metadata.json")
    fingerprint = hashlib.sha256(json.dumps(metadata, sort_keys=True, default=str).encode()).hexdigest()
    mark_success(stage, {"fingerprint": fingerprint, "shape": list(x.shape)})
    return stage
