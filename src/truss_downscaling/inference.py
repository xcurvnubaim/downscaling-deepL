from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from typing import TextIO

import numpy as np
import torch
import xarray as xr

from .checkpointing import mark_success
from .config import Config
from .transforms import to_physical_targets, validate_norm_stats


_WORKER_SOURCE: xr.Dataset | None = None
_WORKER_REGRIDDER = None


def _init_regrid_worker(source_path: str, target_grid_path: str) -> None:
    global _WORKER_SOURCE, _WORKER_REGRIDDER

    import xesmf as xe

    _WORKER_SOURCE = xr.open_dataset(source_path)
    target_grid = xr.open_dataset(target_grid_path)
    _WORKER_REGRIDDER = xe.Regridder(
        _WORKER_SOURCE, target_grid, method="bilinear", periodic=False
    )


def _regrid_channel(task: tuple[str, str, int, int]) -> np.ndarray:
    variable, time_name, start, stop = task
    if _WORKER_SOURCE is None or _WORKER_REGRIDDER is None:
        raise RuntimeError("regridding worker is not initialized")
    values = _WORKER_SOURCE[variable].isel({time_name: slice(start, stop)})
    return _WORKER_REGRIDDER(values).values.astype("float32", copy=False)


def _path(config: Config, key: str) -> Path:
    value = config.data.get(key)
    if not value:
        raise ValueError(f"missing data.{key} in configuration")
    path = Path(value)
    return path if path.is_absolute() else (config.path.parent / path).resolve()


def _resolve_device(config: Config) -> torch.device:
    requested = config.inference.get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested inference device {requested!r}, but CUDA is unavailable")
    return device


def _predict_in_batches(
    model: torch.nn.Module,
    inputs: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("inference.batch_size must be at least 1")
    prediction = np.empty(inputs.shape, dtype="float32")
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            stop = min(start + batch_size, len(inputs))
            batch = torch.from_numpy(inputs[start:stop]).to(device)
            prediction[start:stop] = model(batch).cpu().numpy()
    return prediction


def _versioned_output(config: Config, created_at: datetime) -> tuple[Path, Path]:
    version = created_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = config.output_root / "inference" / version
    name = f"{config.scenario.get('climate_scenario', 'scenario')}_{config.scenario.get('member', 'member')}.nc"
    return output, output / name


def _time_chunks(length: int, chunk_size: int):
    if chunk_size < 1:
        raise ValueError("inference.time_chunk_size must be at least 1")
    for start in range(0, length, chunk_size):
        yield slice(start, min(start + chunk_size, length))


def _cpu_workers(config: Config) -> int:
    workers = int(config.inference.get("cpu_workers", 1))
    if workers < 1:
        raise ValueError("inference.cpu_workers must be at least 1")
    return min(workers, len(config.data["input_channels"]))


def _submit_regrid_chunk(pool, channels: list[str], time_name: str, time_slice: slice):
    return [
        pool.submit(
            _regrid_channel,
            (channel.removesuffix("_gcm"), time_name, time_slice.start, time_slice.stop),
        )
        for channel in channels
    ]


def _collect_regrid_chunk(
    futures,
    time_slice: slice,
    channel_count: int,
    height: int,
    width: int,
    fill_values: np.ndarray,
) -> np.ndarray:
    x = np.empty(
        (time_slice.stop - time_slice.start, channel_count, height, width),
        dtype="float32",
    )
    for index, future in enumerate(futures):
        x[:, index] = future.result()
        np.nan_to_num(x[:, index], nan=float(fill_values[index]), copy=False)
    return x


def _process_ram_gib() -> float | None:
    try:
        resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / 1024**3
    except (OSError, ValueError, IndexError):
        return None


def _process_tree_cpu_time() -> float:
    process_ids = [os.getpid()]
    try:
        children = Path("/proc/thread-self/children").read_text().split()
        process_ids.extend(int(process_id) for process_id in children)
    except (OSError, ValueError):
        pass
    ticks = 0
    for process_id in process_ids:
        try:
            fields = Path(f"/proc/{process_id}/stat").read_text().rsplit(")", 1)[1].split()
            ticks += int(fields[11]) + int(fields[12])
        except (OSError, ValueError, IndexError):
            continue
    return ticks / os.sysconf("SC_CLK_TCK")


class _ProgressBar:
    def __init__(
        self,
        total: int,
        device: torch.device = torch.device("cpu"),
        stream: TextIO = sys.stderr,
    ) -> None:
        self.total = total
        self.device = device
        self.stream = stream
        self.started_at = time.perf_counter()
        self.last_wall_time = self.started_at
        self.last_cpu_time = _process_tree_cpu_time()
        self.nvml = None
        self.gpu_handle = None
        if self.device.type == "cuda":
            try:
                import pynvml

                pynvml.nvmlInit()
                index = self.device.index
                if index is None:
                    index = torch.cuda.current_device()
                properties = torch.cuda.get_device_properties(index)
                uuid = getattr(properties, "uuid", None)
                try:
                    self.gpu_handle = (
                        pynvml.nvmlDeviceGetHandleByUUID(str(uuid)) if uuid is not None else None
                    )
                except (pynvml.NVMLError, TypeError):
                    self.gpu_handle = None
                if self.gpu_handle is None:
                    self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                self.nvml = pynvml
            except Exception:
                self.nvml = None
                self.gpu_handle = None

    def update(self, completed: int) -> None:
        now = time.perf_counter()
        cpu_time = _process_tree_cpu_time()
        elapsed = now - self.started_at
        interval = now - self.last_wall_time
        cpu_percent = (cpu_time - self.last_cpu_time) / interval * 100 if interval else 0.0
        self.last_wall_time = now
        self.last_cpu_time = cpu_time
        fraction = completed / self.total if self.total else 1.0
        width = 30
        filled = min(width, int(width * fraction))
        rate = completed / elapsed if elapsed else 0.0
        remaining = (self.total - completed) / rate if rate else 0.0
        bar = "#" * filled + "-" * (width - filled)
        ram = _process_ram_gib()
        ram_text = f"{ram:.1f}GiB" if ram is not None else "n/a"
        gpu_text = "n/a"
        if self.nvml is not None and self.gpu_handle is not None:
            try:
                utilization = self.nvml.nvmlDeviceGetUtilizationRates(self.gpu_handle).gpu
                memory = self.nvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                gpu_text = f"{utilization}% {memory.used / 1024**3:.1f}/{memory.total / 1024**3:.1f}GiB"
            except self.nvml.NVMLError:
                gpu_text = "n/a"
        end = "\n" if completed >= self.total else ""
        print(
            f"\rInference [{bar}] {completed}/{self.total} "
            f"({fraction:6.2%}) {rate:5.1f} steps/s ETA {remaining:6.0f}s "
            f"CPU {cpu_percent:5.1f}% RAM {ram_text} GPU {gpu_text}",
            end=end,
            file=self.stream,
            flush=True,
        )
        if completed >= self.total and self.nvml is not None:
            self.nvml.nvmlShutdown()
            self.nvml = None
            self.gpu_handle = None


def run_infer(config: Config, force: bool = False, progress: bool = False) -> Path:
    checkpoint_path = _path(config, "checkpoint")
    source_path = _path(config, "gcm_file")
    target_grid_path = _path(config, "target_grid_file")
    if not source_path.exists() or not checkpoint_path.exists() or not target_grid_path.exists():
        raise FileNotFoundError(
            f"inference requires source, target grid, and checkpoint: "
            f"{source_path}, {target_grid_path}, {checkpoint_path}"
        )
    created_at = datetime.now(timezone.utc)
    output, destination = _versioned_output(config, created_at)
    output.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(config)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if int(checkpoint.get("checkpoint_version", 0)) != 2:
        raise RuntimeError("checkpoint uses an older format; retrain after regenerating preprocessing artifacts")
    model = config.model_class()(**config.model.get("parameters", {})).to(device)
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
        time_name = "time" if "time" in ds.coords else "date"
        target_lat = "lat" if "lat" in target_grid.coords else "latitude"
        target_lon = "lon" if "lon" in target_grid.coords else "longitude"
        time_chunk_size = int(config.inference.get("time_chunk_size", 32))
        chunks = list(_time_chunks(ds.sizes[time_name], time_chunk_size))
        batch_size = int(config.inference.get("batch_size", 1))
        cpu_workers = _cpu_workers(config)
        fill_values = np.asarray(norm.get("edge_fill_values", norm["X_mean"]), dtype="float32")
        means = np.asarray(norm["X_mean"], dtype="float32")[None, :, None, None]
        stds = np.asarray(norm["X_std"], dtype="float32")[None, :, None, None]
        coordinates = xr.Dataset(
            coords={
                time_name: ds[time_name],
                target_lat: target_grid[target_lat],
                target_lon: target_grid[target_lon],
            },
        )
        coordinates.to_netcdf(destination, unlimited_dims=[time_name])
        progress_bar = _ProgressBar(ds.sizes[time_name], device=device) if progress else None

        from netCDF4 import Dataset

        pool = ProcessPoolExecutor(
            max_workers=cpu_workers,
            mp_context=get_context("spawn"),
            initializer=_init_regrid_worker,
            initargs=(str(source_path), str(target_grid_path)),
        )
        with pool, Dataset(destination, "a") as result:
            output_variables = {
                channel.removesuffix("_era5"): result.createVariable(
                    channel.removesuffix("_era5"),
                    "f4",
                    (time_name, target_lat, target_lon),
                    zlib=True,
                    complevel=4,
                    chunksizes=(
                        min(time_chunk_size, ds.sizes[time_name]),
                        target_grid.sizes[target_lat],
                        target_grid.sizes[target_lon],
                    ),
                )
                for channel in config.data["target_channels"]
            }
            chunk_iterator = iter(chunks)
            time_slice = next(chunk_iterator, None)
            futures = (
                _submit_regrid_chunk(pool, config.data["input_channels"], time_name, time_slice)
                if time_slice is not None
                else []
            )
            while time_slice is not None:
                x = _collect_regrid_chunk(
                    futures,
                    time_slice,
                    len(config.data["input_channels"]),
                    target_grid.sizes[target_lat],
                    target_grid.sizes[target_lon],
                    fill_values,
                )
                next_slice = next(chunk_iterator, None)
                next_futures = (
                    _submit_regrid_chunk(
                        pool, config.data["input_channels"], time_name, next_slice
                    )
                    if next_slice is not None
                    else []
                )
                if norm.get("log1p_input_applied"):
                    pr_index = int(norm["pr_index"])
                    x[:, pr_index] = np.log1p(np.maximum(x[:, pr_index], 0))
                x -= means
                x /= stds
                prediction = _predict_in_batches(model, x, batch_size, device)
                prediction = to_physical_targets(prediction, norm)
                for index, channel in enumerate(config.data["target_channels"]):
                    variable = channel.removesuffix("_era5")
                    if variable == "hurs":
                        prediction[:, index] = np.clip(prediction[:, index], 0, 100)
                    elif variable == "pr":
                        prediction[:, index] = np.maximum(prediction[:, index], 0)
                    output_variables[variable][time_slice, :, :] = prediction[:, index]
                if progress_bar is not None:
                    progress_bar.update(time_slice.stop)
                time_slice = next_slice
                futures = next_futures
    mark_success(
        output,
        {
            "file": str(destination),
            "version": output.name,
            "created_at": created_at.isoformat(),
            "scenario": config.scenario,
            "source": str(source_path),
            "target_grid": str(target_grid_path),
            "checkpoint": str(checkpoint_path),
            "device": str(device),
            "batch_size": batch_size,
            "time_chunk_size": time_chunk_size,
            "cpu_workers": cpu_workers,
            "regridding": "xesmf_bilinear",
            "precipitation_inverse_transform": "expm1_clipped_at_zero",
        },
    )
    return destination
