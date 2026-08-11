from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
import torch

from .checkpointing import mark_success
from .config import Config
from .models.residual_unet import ResidualUNet


def run_infer(config: Config, force: bool = False) -> Path:
    checkpoint_path = Path(config.data["checkpoint"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = (config.path.parent / checkpoint_path).resolve()
    source_path = Path(config.data["gcm_file"])
    if not source_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(f"inference requires source and checkpoint: {source_path}, {checkpoint_path}")
    output = config.output_root / "inference"
    output.mkdir(parents=True, exist_ok=True)
    name = f"{config.scenario.get('climate_scenario', 'scenario')}_{config.scenario.get('member', 'member')}.nc"
    destination = output / name
    if destination.exists() and not force:
        return destination
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = config.model_class()(**config.model.get("parameters", {}))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    norm = checkpoint.get("norm_stats")
    if norm is None:
        norm_path = checkpoint_path.parent.parent / "preprocessing" / "normalization.json"
        if not norm_path.exists():
            raise ValueError("checkpoint must contain norm_stats or have sibling preprocessing normalization.json")
        import json
        norm = json.loads(norm_path.read_text(encoding="utf-8"))
    target_grid_path = Path(config.data["target_grid_file"])
    if not target_grid_path.is_absolute():
        target_grid_path = (config.path.parent / target_grid_path).resolve()
    with xr.open_dataset(source_path) as ds, xr.open_dataset(target_grid_path) as target_grid:
        lat_name = "lat" if "lat" in ds.coords else "latitude"
        lon_name = "lon" if "lon" in ds.coords else "longitude"
        time_name = "time" if "time" in ds.coords else "date"
        target_lat = "lat" if "lat" in target_grid.coords else "latitude"
        target_lon = "lon" if "lon" in target_grid.coords else "longitude"
        arrays = []
        for channel in config.data["input_channels"]:
            variable = channel.removesuffix("_gcm")
            arrays.append(ds[variable].interp({lat_name: target_grid[target_lat], lon_name: target_grid[target_lon]}).values.astype("float32"))
        x = np.stack(arrays, axis=1)
        if norm.get("log1p_applied"):
            pr_index = int(norm.get("pr_index", 2)); x[:, pr_index] = np.log1p(np.maximum(x[:, pr_index], 0))
        means = np.asarray(norm["X_mean"], dtype="float32")[None, :, None, None]
        stds = np.asarray(norm["X_std"], dtype="float32")[None, :, None, None]
        with torch.no_grad():
            prediction = model(torch.from_numpy((x - means) / stds)).numpy()
        y_means = np.asarray(norm["Y_mean"], dtype="float32")[None, :, None, None]
        y_stds = np.asarray(norm["Y_std"], dtype="float32")[None, :, None, None]
        prediction = prediction * y_stds + y_means
        prediction[:, 1] = np.clip(prediction[:, 1], 0, 100)
        prediction[:, 2] = np.maximum(prediction[:, 2], 0)
        result = xr.Dataset(
            {channel.removesuffix("_era5"): ((time_name, target_lat, target_lon), prediction[:, i]) for i, channel in enumerate(config.data["target_channels"])},
            coords={time_name: ds[time_name], target_lat: target_grid[target_lat], target_lon: target_grid[target_lon]},
        )
        result.to_netcdf(destination)
    mark_success(output, {"file": str(destination), "scenario": config.scenario, "checkpoint": str(checkpoint_path)})
    return destination
