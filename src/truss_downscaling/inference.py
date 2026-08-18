from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from .checkpointing import mark_success
from .config import Config
from .transforms import to_physical_targets, validate_norm_stats


def _path(config: Config, key: str) -> Path:
    value = config.data.get(key)
    if not value:
        raise ValueError(f"missing data.{key} in configuration")
    path = Path(value)
    return path if path.is_absolute() else (config.path.parent / path).resolve()


def run_infer(config: Config, force: bool = False) -> Path:
    checkpoint_path = _path(config, "checkpoint")
    source_path = _path(config, "gcm_file")
    target_grid_path = _path(config, "target_grid_file")
    if not source_path.exists() or not checkpoint_path.exists() or not target_grid_path.exists():
        raise FileNotFoundError(
            f"inference requires source, target grid, and checkpoint: "
            f"{source_path}, {target_grid_path}, {checkpoint_path}"
        )
    output = config.output_root / "inference"
    output.mkdir(parents=True, exist_ok=True)
    name = f"{config.scenario.get('climate_scenario', 'scenario')}_{config.scenario.get('member', 'member')}.nc"
    destination = output / name
    if destination.exists() and not force:
        return destination

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("checkpoint_version", 0)) != 2:
        raise RuntimeError("checkpoint uses an older format; retrain after regenerating preprocessing artifacts")
    model = config.model_class()(**config.model.get("parameters", {}))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    norm = checkpoint.get("norm_stats")
    if norm is None:
        norm_path = checkpoint_path.parent.parent / "preprocessing" / "normalization.json"
        if not norm_path.exists():
            raise ValueError("checkpoint must contain norm_stats or have sibling preprocessing normalization.json")
        norm = json.loads(norm_path.read_text(encoding="utf-8"))
    validate_norm_stats(norm)

    try:
        import xesmf as xe
    except ImportError as error:
        raise RuntimeError(
            "inference requires xESMF bilinear regridding; install the data extra"
        ) from error

    with xr.open_dataset(source_path) as ds, xr.open_dataset(target_grid_path) as target_grid:
        lat_name = "lat" if "lat" in ds.coords else "latitude"
        lon_name = "lon" if "lon" in ds.coords else "longitude"
        time_name = "time" if "time" in ds.coords else "date"
        target_lat = "lat" if "lat" in target_grid.coords else "latitude"
        target_lon = "lon" if "lon" in target_grid.coords else "longitude"
        regridder = xe.Regridder(ds, target_grid, method="bilinear", periodic=False)
        arrays = []
        for channel in config.data["input_channels"]:
            variable = channel.removesuffix("_gcm")
            mapped = regridder(ds[variable]).values.astype("float32")
            arrays.append(mapped)
        x = np.stack(arrays, axis=1)
        if norm.get("log1p_input_applied"):
            pr_index = int(norm["pr_index"])
            x[:, pr_index] = np.log1p(np.maximum(x[:, pr_index], 0))
        fill_values = np.asarray(norm.get("edge_fill_values", norm["X_mean"]), dtype="float32")
        for channel in range(x.shape[1]):
            x[:, channel] = np.nan_to_num(x[:, channel], nan=float(fill_values[channel]))
        means = np.asarray(norm["X_mean"], dtype="float32")[None, :, None, None]
        stds = np.asarray(norm["X_std"], dtype="float32")[None, :, None, None]
        with torch.no_grad():
            prediction = model(torch.from_numpy((x - means) / stds)).numpy()
        prediction = to_physical_targets(prediction, norm)
        for index, channel in enumerate(config.data["target_channels"]):
            variable = channel.removesuffix("_era5")
            if variable == "hurs":
                prediction[:, index] = np.clip(prediction[:, index], 0, 100)
            elif variable == "pr":
                prediction[:, index] = np.maximum(prediction[:, index], 0)
        result = xr.Dataset(
            {
                channel.removesuffix("_era5"): ((time_name, target_lat, target_lon), prediction[:, index])
                for index, channel in enumerate(config.data["target_channels"])
            },
            coords={
                time_name: ds[time_name],
                target_lat: target_grid[target_lat],
                target_lon: target_grid[target_lon],
            },
        )
        result.to_netcdf(destination)
    mark_success(
        output,
        {
            "file": str(destination),
            "scenario": config.scenario,
            "checkpoint": str(checkpoint_path),
            "regridding": "xesmf_bilinear",
            "precipitation_inverse_transform": "expm1_clipped_at_zero",
        },
    )
    return destination
