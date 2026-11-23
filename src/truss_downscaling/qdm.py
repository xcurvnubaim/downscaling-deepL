"""Terminal QDM (quantile delta mapping) marginal finalizer.

The last operator in the pipeline: applied to the CNN's fine-grid output, never
before it and never alone. For each pixel the finalizer learns the model's own
error distribution against native ERA5 over a fit window, then applies that
correction to everything the model produces.

What it corrects, per pixel:

- ``pr``  : multiplicative delta on quantile ratios, so the correction scales with
            the value instead of adding a fixed offset; wet-day occurrence is
            restored by deterministic within-mass frequency adaptation (pure
            multiplicative QDM maps ``v = 0 -> 0`` and preserves dry-day counts
            exactly, so occurrence must be corrected explicitly); tails are
            clamped to the fit-window per-cell ERA5 maximum.
- ``tasmax``/``hurs`` : additive delta (offset), removing passed-through level
            biases; ``hurs`` is clipped to ``[0, 100]``.

What it does not correct: day-to-day phase, spatial texture, or the model's own
structure — those belong to the model stages, and treating them as finalizer
problems is how the earlier QM-on-CNN experiment failed.

The fitted artifact is per member and is applied unchanged to future periods
(no truth exists there; the transfer assumption is recorded in the artifact
manifest) and identically to both scenario sides of any comparison, so the
correction cancels in differences.

Numerics: the per-cell search is vectorised with column offsets. ``_OFFSET`` is
a power of two chosen so every ``value + column * _OFFSET`` sum stays below
``2**28``; float64 then resolves every float32 mantissa bit exactly, making the
vectorised search bit-identical to a per-cell loop (verified in tests).

Implementation choice (the F0 decision from the QDM plan, recorded here): the
core is a small self-contained numpy implementation rather than ``xclim.sdba``
because (a) this project pins exact dependency versions and the finalizer runs
inside Kaggle kernels, where xclim's transitive dependency tree is a
version-drift risk on the pinned stack; (b) the wet-day frequency adaptation
must be deterministic and chunk-invariant — a fixed day/cell hash rather than
random jitter — so every run and artifact is reproducible bit-for-bit; and
(c) the mapping plugs directly into the streaming netCDF writer that inference
uses. The accuracy burden of a hand-rolled implementation is paid once and
kept automated: ``tests/test_qdm.py`` asserts bit-identity against a naive
per-cell reference and recovery of synthetic distortions, and the 2026-11-22
Kaggle pilot reproduced the frozen evaluation harness exactly. Swapping the
core for ``xclim.sdba`` behind this same stage interface remains possible if
a library-backed implementation is preferred.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from .checkpointing import mark_success, save_json, stage_is_complete
from .config import Config

_ALLOWED_VARIABLES = ("pr", "tasmax", "hurs")
_DELTA_MODE = {"pr": "ratio", "tasmax": "offset", "hurs": "offset"}
_HURS_BOUNDS = (0.0, 100.0)
_COLUMN_BLOCK = 4096
_OFFSET = 2.0**13
_QM_GUARD_MM = 0.1


def _cell_quantile(block: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Linearly interpolated per-cell quantiles; ``block`` is (n, C), ``p`` is (T, C)."""
    index = p * (block.shape[0] - 1)
    lo = np.clip(index.astype(np.int64), 0, block.shape[0] - 1)
    hi = np.clip(lo + 1, 0, block.shape[0] - 1)
    fraction = index - lo
    lower = np.take_along_axis(block, lo, axis=0)
    upper = np.take_along_axis(block, hi, axis=0)
    return lower + fraction * (upper - lower)


def map_cells(
    values: np.ndarray,
    model_sorted: np.ndarray,
    truth_sorted: np.ndarray,
    variable: str,
    day_indices: np.ndarray,
    wet_threshold: float = 1.0,
    tail_clamp: bool = True,
    column_block: int = _COLUMN_BLOCK,
) -> tuple[np.ndarray, dict[str, int]]:
    """Map per-cell values through the fit-window distributions.

    ``values`` is (T, C) float32; ``model_sorted``/``truth_sorted`` are (n, C)
    float32, sorted along axis 0; ``day_indices`` carries the global day index of
    each row and drives the deterministic zero-jitter (so chunked application is
    identical to a single call). Returns the mapped (T, C) float32 array and
    occurrence/tail statistics.
    """
    if variable not in _DELTA_MODE:
        raise ValueError(f"unsupported finalizer variable {variable!r}; expected one of {sorted(_ALLOWED_VARIABLES)}")
    n = model_sorted.shape[0]
    if truth_sorted.shape != model_sorted.shape:
        raise ValueError("map_cells: model and truth fit arrays must share the (n, C) shape")
    if values.ndim != 2 or values.shape[1] != model_sorted.shape[1]:
        raise ValueError("map_cells: values (T, C) must share the cell axis with the fit arrays")
    if len(day_indices) != values.shape[0]:
        raise ValueError("map_cells: one day index is required per value row")
    if column_block < 1:
        raise ValueError("column_block must be at least 1")
    for array in (values, model_sorted, truth_sorted):
        if not np.isfinite(array).all():
            raise ValueError(f"{variable}: non-finite values reached the finalizer")
        if float(np.abs(array).max()) >= _OFFSET:
            raise ValueError(f"{variable}: value outside the exact-offset search range")

    total_rows, cells = values.shape
    out = np.empty((total_rows, cells), dtype=np.float32)
    stats = {"clamped": 0, "wet_invented": 0, "wet_removed": 0}
    mode = _DELTA_MODE[variable]
    day_idx = np.asarray(day_indices, dtype=np.int64)[:, None]
    for b0 in range(0, cells, column_block):
        cs = slice(b0, min(b0 + column_block, cells))
        m_blk = model_sorted[:, cs].astype(np.float64)
        e_blk = truth_sorted[:, cs].astype(np.float64)
        v_blk = values[:, cs].astype(np.float64)
        width = m_blk.shape[1]

        column_offsets = np.arange(width, dtype=np.float64) * _OFFSET
        flat_model = (m_blk + column_offsets[None, :]).ravel(order="F")
        positions = np.searchsorted(flat_model, (v_blk + column_offsets[None, :]).ravel(), side="right")
        positions = positions.reshape(v_blk.shape) - (np.arange(width) * n)[None, :]
        p = np.clip(positions / n, 0.0, 1.0)

        # Frequency adaptation: exact zeros are tied, so they receive a
        # deterministic hash position spread over the model's own dry mass.
        # Dry days below ERA5's dry fraction stay dry; the rest take ERA5's
        # low wet quantiles. The too-wet direction needs no help.
        if variable == "pr":
            dry_mass = (m_blk == 0.0).sum(axis=0) / n
            cell_hash = day_idx * np.int64(2654435761) + (b0 + np.arange(width, dtype=np.int64))[None, :] * np.int64(40503)
            jitter = (cell_hash % np.int64(4294967296)).astype(np.float64) / 4294967296.0
            p = np.where(v_blk <= 0.0, dry_mass[None, :] * jitter, p)

        model_quantile = _cell_quantile(m_blk, p)
        truth_quantile = _cell_quantile(e_blk, p)
        if mode == "ratio":
            wet = v_blk >= wet_threshold
            block_out = np.where(
                wet & (model_quantile >= _QM_GUARD_MM),
                truth_quantile * v_blk / np.maximum(model_quantile, 1e-6),
                truth_quantile + (v_blk - model_quantile),
            )
        else:
            block_out = truth_quantile + (v_blk - model_quantile)

        if variable == "pr":
            stats["wet_invented"] += int(((v_blk <= 0.0) & (block_out >= wet_threshold)).sum())
            stats["wet_removed"] += int(((v_blk >= wet_threshold) & (block_out < wet_threshold)).sum())
            if tail_clamp:
                truth_max = e_blk[-1]
                stats["clamped"] += int((block_out > truth_max[None, :]).sum())
                block_out = np.minimum(block_out, truth_max[None, :])
            block_out = np.maximum(block_out, 0.0)
        if variable == "hurs":
            block_out = np.clip(block_out, *_HURS_BOUNDS)
        out[:, cs] = block_out.astype(np.float32)
    return out, stats


def fit_finalize(model_cells: np.ndarray, truth_cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build the per-cell fit arrays: both (n, C) float32, sorted along axis 0."""
    model_cells = np.ascontiguousarray(model_cells, dtype=np.float32)
    truth_cells = np.ascontiguousarray(truth_cells, dtype=np.float32)
    if model_cells.ndim != 2 or model_cells.shape != truth_cells.shape:
        raise ValueError("fit_finalize: model and truth must share the same (n, C) shape")
    if model_cells.shape[0] < 2:
        raise ValueError("fit_finalize: at least two fit days are required")
    return np.sort(model_cells, axis=0), np.sort(truth_cells, axis=0)


def core_mask(lat: np.ndarray, lon: np.ndarray, source_lat: np.ndarray, source_lon: np.ndarray) -> np.ndarray:
    """Fine-grid cells whose centres fall inside the coarse driver's bounding box.

    The finalizer corrects these core cells; the edge buffer keeps the raw model
    values, matching the frozen evaluation contract.
    """
    inside_lat = (np.asarray(lat, dtype=np.float64) >= source_lat.min()) & (
        np.asarray(lat, dtype=np.float64) <= source_lat.max()
    )
    inside_lon = (np.asarray(lon, dtype=np.float64) >= source_lon.min()) & (
        np.asarray(lon, dtype=np.float64) <= source_lon.max()
    )
    return inside_lat[:, None] & inside_lon[None, :]


def _to_cells(field: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(field[:, rows, cols])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _coord_names(dataset: xr.Dataset) -> tuple[str, str, str]:
    time_name = "time" if "time" in dataset.coords else "date"
    lat_name = "lat" if "lat" in dataset.coords else "latitude"
    lon_name = "lon" if "lon" in dataset.coords else "longitude"
    return time_name, lat_name, lon_name


def _resolve_path(config: Config, value: str | Path | None, key: str) -> Path:
    if not value:
        raise ValueError(f"missing path for {key}")
    path = Path(value)
    return path if path.is_absolute() else (config.path.parent / path).resolve()


def _qdm_input_path(config: Config) -> Path:
    override = config.qdm.get("input_file")
    if override:
        return _resolve_path(config, override, "qdm.input_file")
    root = config.output_root / "inference"
    if not root.exists():
        raise FileNotFoundError(
            "qdm.input_file is unset and no inference output exists; "
            "run the infer stage first or set qdm.input_file"
        )
    candidates = sorted(path for path in root.iterdir() if stage_is_complete(path))
    if not candidates:
        raise RuntimeError("no completed inference version found; rerun the infer stage")
    version = candidates[-1]
    name = f"{config.scenario.get('climate_scenario', 'scenario')}_{config.scenario.get('member', 'member')}.nc"
    destination = version / name
    if not destination.exists():
        netcdf_files = sorted(version.glob("*.nc"))
        if len(netcdf_files) != 1:
            raise FileNotFoundError(f"cannot locate the inference NetCDF in {version}")
        destination = netcdf_files[0]
    return destination


def _settings(config: Config) -> dict[str, Any]:
    block = config.qdm
    variables = tuple(block.get("variables", ()))
    if not variables:
        raise ValueError("qdm.variables is empty; name at least one of pr/tasmax/hurs")
    unknown = [name for name in variables if name not in _ALLOWED_VARIABLES]
    if unknown:
        raise ValueError(f"qdm.variables contains unsupported entries: {unknown}")
    if len(set(variables)) != len(variables):
        raise ValueError("qdm.variables must not repeat a variable")
    fit_window = block.get("fit_window")
    if not (isinstance(fit_window, (list, tuple)) and len(fit_window) == 2 and all(fit_window)):
        raise ValueError("qdm.fit_window must be [start, end] as ISO date strings")
    wet_threshold = float(block.get("wet_threshold_mm", 1.0))
    if wet_threshold <= 0:
        raise ValueError("qdm.wet_threshold_mm must be positive")
    chunk = int(block.get("time_chunk_size", 128))
    if chunk < 1:
        raise ValueError("qdm.time_chunk_size must be at least 1")
    return {
        "variables": variables,
        "fit_start": str(fit_window[0]),
        "fit_end": str(fit_window[1]),
        "wet_threshold_mm": wet_threshold,
        "tail_clamp": bool(block.get("tail_clamp", True)),
        "time_chunk_size": chunk,
    }


def _versioned_output(config: Config, created_at: datetime) -> tuple[Path, Path]:
    version = created_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = config.output_root / "finalize" / version
    name = f"{config.scenario.get('climate_scenario', 'scenario')}_{config.scenario.get('member', 'member')}_finalized.nc"
    return output, output / name


def run_finalize(config: Config, force: bool = False) -> Path:  # noqa: ARG001 (mirrors the stage signatures)
    """Fit the finalizer on the configured window and apply it to the whole input."""
    settings = _settings(config)
    input_path = _qdm_input_path(config)
    truth_path = _resolve_path(config, config.qdm.get("truth_file") or config.data.get("target_grid_file"), "truth file")
    mask_path = _resolve_path(config, config.qdm.get("mask_file") or config.data.get("gcm_file"), "mask grid file")
    if not input_path.exists():
        raise FileNotFoundError(f"finalize requires the model output: {input_path}")
    for path, label in ((truth_path, "truth"), (mask_path, "mask grid")):
        if not path.exists():
            raise FileNotFoundError(f"finalize requires the {label} file: {path}")

    created_at = datetime.now(timezone.utc)
    output, destination = _versioned_output(config, created_at)
    output.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(input_path) as source, xr.open_dataset(truth_path) as truth, xr.open_dataset(mask_path) as mask_source:
        time_name, lat_name, lon_name = _coord_names(source)
        truth_time_name, truth_lat_name, truth_lon_name = _coord_names(truth)
        _, mask_lat_name, mask_lon_name = _coord_names(mask_source)
        source_lat = source[lat_name].values.astype(np.float64)
        source_lon = source[lon_name].values.astype(np.float64)
        for name, actual, expected in (
            (truth_lat_name, truth[truth_lat_name].values.astype(np.float64), source_lat),
            (truth_lon_name, truth[truth_lon_name].values.astype(np.float64), source_lon),
        ):
            if actual.shape != expected.shape or not np.allclose(actual, expected):
                raise ValueError(f"truth grid {name} differs from the model output grid")
        mask = core_mask(
            source_lat,
            source_lon,
            mask_source[mask_lat_name].values.astype(np.float64),
            mask_source[mask_lon_name].values.astype(np.float64),
        )
        rows, cols = np.where(mask)
        core_cells = int(mask.sum())

        source_time = source[time_name].values
        truth_time = truth[truth_time_name].values
        fit_idx = np.where((source_time >= np.datetime64(settings["fit_start"])) & (source_time <= np.datetime64(settings["fit_end"])))[0]
        if fit_idx.size < 2:
            raise ValueError("the fit window selects fewer than two days in the model output")
        if not np.isin(source_time[fit_idx], truth_time).all():
            raise RuntimeError("the truth file does not cover every fit-window day")

        model_fit: dict[str, np.ndarray] = {}
        truth_fit: dict[str, np.ndarray] = {}
        for variable in settings["variables"]:
            model_cells = _to_cells(source[variable].isel({time_name: fit_idx}).values.astype(np.float32), rows, cols)
            truth_cells = _to_cells(truth[variable].isel({truth_time_name: fit_idx}).values.astype(np.float32), rows, cols)
            model_fit[variable], truth_fit[variable] = fit_finalize(model_cells, truth_cells)
            print(f"fitted {variable}: {fit_idx.size} days x {core_cells} core cells", flush=True)

        fit_path = output / "qdm_fit.npz"
        np.savez_compressed(fit_path, **{f"{variable}__model": model_fit[variable] for variable in settings["variables"]}, **{f"{variable}__truth": truth_fit[variable] for variable in settings["variables"]})
        fit_manifest = {
            "variables": list(settings["variables"]),
            "fit_window": [settings["fit_start"], settings["fit_end"]],
            "n_fit_days": int(fit_idx.size),
            "core_cells": core_cells,
            "wet_threshold_mm": settings["wet_threshold_mm"],
            "delta_mode": {variable: _DELTA_MODE[variable] for variable in settings["variables"]},
            "tail_clamp": settings["tail_clamp"],
            "model_source": str(input_path),
            "truth_source": str(truth_path),
            "artifact": str(fit_path),
            "artifact_sha256": _sha256(fit_path),
            "transfer_assumption": (
                "the fitted correction is applied unchanged to future periods (no truth exists there) "
                "and identically to both scenario sides of any comparison, so it cancels in differences"
            ),
        }
        save_json(fit_manifest, output / "qdm_fit_manifest.json")

        data_variables = [name for name in source.data_vars]
        finalized_variables = set(settings["variables"])
        copied_variables = [name for name in data_variables if name not in finalized_variables]
        total_days = source.sizes[time_name]
        chunk_size = settings["time_chunk_size"]
        stats = {variable: {"clamped": 0, "wet_invented": 0, "wet_removed": 0} for variable in settings["variables"]}
        coordinates = xr.Dataset(
            coords={
                time_name: source[time_name],
                lat_name: source[lat_name],
                lon_name: source[lon_name],
            }
        )
        coordinates.to_netcdf(destination, unlimited_dims=[time_name])

        from netCDF4 import Dataset as NetCDFDataset

        with NetCDFDataset(destination, "a") as result:
            out_variables = {
                name: result.createVariable(
                    name,
                    "f4",
                    (time_name, lat_name, lon_name),
                    zlib=True,
                    complevel=4,
                    chunksizes=(min(chunk_size, total_days), source.sizes[lat_name], source.sizes[lon_name]),
                )
                for name in data_variables
            }
            for start in range(0, total_days, chunk_size):
                stop = min(start + chunk_size, total_days)
                time_slice = slice(start, stop)
                day_indices = np.arange(start, stop)
                for variable in settings["variables"]:
                    raw = source[variable].isel({time_name: time_slice}).values.astype(np.float32)
                    mapped, chunk_stats = map_cells(
                        _to_cells(raw, rows, cols),
                        model_fit[variable],
                        truth_fit[variable],
                        variable,
                        day_indices,
                        wet_threshold=settings["wet_threshold_mm"],
                        tail_clamp=settings["tail_clamp"],
                    )
                    corrected = raw.copy()
                    corrected[:, rows, cols] = mapped
                    out_variables[variable][time_slice, :, :] = corrected
                    for key, value in chunk_stats.items():
                        stats[variable][key] += value
                for variable in copied_variables:
                    raw = source[variable].isel({time_name: time_slice}).values.astype(np.float32)
                    out_variables[variable][time_slice, :, :] = raw
                print(f"finalize [{min(stop, total_days)}/{total_days}] days written", flush=True)

    mark_success(
        output,
        {
            "file": str(destination),
            "version": output.name,
            "created_at": created_at.isoformat(),
            "scenario": config.scenario,
            "model_source": str(input_path),
            "truth_source": str(truth_path),
            "mask_source": str(mask_path),
            "settings": settings,
            "occurrence_and_tail_stats": stats,
            "fit_manifest": str(output / "qdm_fit_manifest.json"),
        },
    )
    return destination
