"""Synthetic self-tests for the QDM marginal finalizer.

No climate data is required: every fixture is generated with a known distortion
(mean shift, variance compression, dry-day injection) that the finalizer must
recover. The per-cell reference loop proves the vectorised search is exact.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from truss_downscaling import cli
from truss_downscaling.config import load_config
from truss_downscaling.qdm import core_mask, fit_finalize, map_cells, run_finalize

WET_MM = 1.0


def _naive_map(values, model_sorted, truth_sorted, day_indices, wet_threshold=WET_MM):
    """Per-cell reference: the same math, one slow loop."""
    n, cells = model_sorted.shape
    out = np.empty_like(values, dtype=np.float32)
    clamped = 0
    for cell in range(cells):
        ms = model_sorted[:, cell].astype(np.float64)
        es = truth_sorted[:, cell].astype(np.float64)
        vv = values[:, cell].astype(np.float64)
        positions = np.searchsorted(ms, vv, side="right")
        p = np.clip(positions / n, 0.0, 1.0)
        dry_mass = float((ms == 0.0).sum()) / n
        for row in range(vv.shape[0]):
            if vv[row] <= 0.0:
                cell_hash = (np.int64(day_indices[row]) * np.int64(2654435761) + np.int64(cell) * np.int64(40503)) % np.int64(4294967296)
                p[row] = dry_mass * (float(cell_hash) / 4294967296.0)
        index = p * (n - 1)
        lo = np.clip(index.astype(np.int64), 0, n - 1)
        hi = np.clip(lo + 1, 0, n - 1)
        fraction = index - lo
        qm = ms[lo] + fraction * (ms[hi] - ms[lo])
        qe = es[lo] + fraction * (es[hi] - es[lo])
        mapped = np.where((vv >= wet_threshold) & (qm >= 0.1), qe * vv / np.maximum(qm, 1e-6), qe + (vv - qm))
        clamped += int((mapped > es[-1]).sum())
        mapped = np.maximum(np.minimum(mapped, es[-1]), 0.0)
        out[:, cell] = mapped.astype(np.float32)
    return out, clamped


def test_map_cells_is_bit_identical_to_the_naive_per_cell_reference():
    rng = np.random.default_rng(7)
    n, days, cells = 500, 90, 3000
    model = rng.gamma(2.0, 3.0, size=(n, cells)).astype(np.float32)
    model[rng.random(model.shape) < 0.2] = 0.0
    truth = rng.gamma(2.0, 3.0, size=(n, cells)).astype(np.float32)
    truth[rng.random(truth.shape) < 0.15] = 0.0
    model_sorted, truth_sorted = fit_finalize(model, truth)
    values = rng.gamma(2.0, 3.0, size=(days, cells)).astype(np.float32)
    values[rng.random(values.shape) < 0.25] = 0.0
    values[0, 0] = 5000.0

    # column_block=1000 forces multiple vectorised blocks.
    day_indices = np.arange(1000, 1000 + days)
    mapped, stats = map_cells(values, model_sorted, truth_sorted, "pr", day_indices, column_block=1000)
    reference, reference_clamped = _naive_map(values, model_sorted, truth_sorted, day_indices)

    assert np.array_equal(mapped, reference)
    assert stats["clamped"] == reference_clamped


def test_map_cells_is_chunk_invariant():
    rng = np.random.default_rng(11)
    n, days, cells = 200, 40, 500
    model = rng.gamma(2.0, 3.0, size=(n, cells)).astype(np.float32)
    model[rng.random(model.shape) < 0.3] = 0.0
    truth = rng.gamma(2.0, 3.0, size=(n, cells)).astype(np.float32)
    model_sorted, truth_sorted = fit_finalize(model, truth)
    values = rng.gamma(2.0, 3.0, size=(days, cells)).astype(np.float32)
    values[rng.random(values.shape) < 0.35] = 0.0

    whole, _ = map_cells(values, model_sorted, truth_sorted, "pr", np.arange(50, 50 + days))
    first, _ = map_cells(values[:25], model_sorted, truth_sorted, "pr", np.arange(50, 75))
    second, _ = map_cells(values[25:], model_sorted, truth_sorted, "pr", np.arange(75, 90))

    assert np.array_equal(whole[:25], first)
    assert np.array_equal(whole[25:], second)


def test_pr_recovery_closes_bias_variance_and_occurrence():
    rng = np.random.default_rng(7)
    fit_days, eval_days, cells = 2922, 731, 400
    truth = rng.gamma(2.0, 3.0, size=(fit_days + eval_days, cells)).astype(np.float32)
    truth[rng.random(truth.shape) < 0.18] = 0.0
    model = 0.64 * truth - 1.4
    model[model < 1.0] = 0.0

    model_sorted, truth_sorted = fit_finalize(model[:fit_days], truth[:fit_days])
    mapped, stats = map_cells(model[fit_days:], model_sorted, truth_sorted, "pr", np.arange(eval_days))
    target = truth[fit_days:]

    assert abs(float(mapped.mean() - target.mean())) <= 0.02 * target.mean()
    assert 0.9 <= float(mapped.std() / target.std()) <= 1.1
    assert abs(float((mapped >= WET_MM).mean() - (target >= WET_MM).mean())) <= 0.03
    assert stats["wet_invented"] > 0


def test_offset_mode_removes_affine_bias():
    rng = np.random.default_rng(3)
    fit_days, eval_days, cells = 2922, 365, 300
    truth = rng.normal(30.0, 4.0, size=(fit_days + eval_days, cells)).astype(np.float32)
    model = (0.9 * truth + 3.0).astype(np.float32)

    model_sorted, truth_sorted = fit_finalize(model[:fit_days], truth[:fit_days])
    mapped, _ = map_cells(model[fit_days:], model_sorted, truth_sorted, "tasmax", np.arange(eval_days))

    assert abs(float(mapped.mean() - truth[fit_days:].mean())) < 0.15


def test_hurs_is_clipped_to_physical_bounds():
    rng = np.random.default_rng(5)
    fit_days, eval_days, cells = 500, 100, 50
    truth = rng.normal(55.0, 10.0, size=(fit_days + eval_days, cells)).astype(np.float32)
    model = (1.6 * truth - 20.0).astype(np.float32)

    model_sorted, truth_sorted = fit_finalize(model[:fit_days], truth[:fit_days])
    mapped, _ = map_cells(model[fit_days:], model_sorted, truth_sorted, "hurs", np.arange(eval_days))

    assert mapped.min() >= 0.0
    assert mapped.max() <= 100.0


def test_tail_clamp_caps_extremes_and_counts():
    rng = np.random.default_rng(9)
    fit_days, cells = 500, 100
    truth = rng.gamma(2.0, 3.0, size=(fit_days, cells)).astype(np.float32)
    model = truth.copy()
    model_sorted, truth_sorted = fit_finalize(model, truth)
    values = np.full((3, cells), 5000.0, dtype=np.float32)

    mapped, stats = map_cells(values, model_sorted, truth_sorted, "pr", np.arange(3))

    assert stats["clamped"] == 3 * cells
    assert float(mapped.max()) <= float(truth.max())


def test_map_cells_rejects_bad_shapes_and_unknown_variables():
    rng = np.random.default_rng(1)
    model_sorted, truth_sorted = fit_finalize(rng.gamma(2.0, 2.0, size=(10, 4)).astype(np.float32), rng.gamma(2.0, 2.0, size=(10, 4)).astype(np.float32))
    values = np.zeros((3, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        map_cells(values, model_sorted, truth_sorted, "heatindex", np.arange(3))
    with pytest.raises(ValueError):
        map_cells(values, model_sorted, truth_sorted[:, :3], "pr", np.arange(3))
    with pytest.raises(ValueError):
        map_cells(values, model_sorted, truth_sorted, "pr", np.arange(2))


def test_core_mask_covers_the_coarse_bounding_box():
    lat = np.arange(9.0)
    lon = np.arange(11.0)
    mask = core_mask(lat, lon, np.array([1.0, 6.0]), np.array([2.0, 8.0]))
    assert int(mask.sum()) == 6 * 7
    assert mask[0, :].sum() == 0 and mask[8, :].sum() == 0
    assert mask[:, 0].sum() == 0 and mask[:, 10].sum() == 0


def _write_synthetic_netcdf(path: Path, values: np.ndarray, lat: np.ndarray, lon: np.ndarray, dates: np.ndarray, variable: str = "pr") -> None:
    xr.Dataset({variable: (("time", "lat", "lon"), values)}, coords={"time": dates, "lat": lat, "lon": lon}).to_netcdf(path)


def _smoke_qdm_config(tmp_path: Path, model_file: Path, truth_file: Path, coarse_file: Path, extra_qdm: str = "") -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "run:\n"
        "  id: qdm-smoke\n"
        "  output_root: %s\n"
        "scenario:\n"
        "  mode: inference\n"
        "  climate_scenario: ssp245\n"
        "  member: r1i1p1f2\n"
        "data:\n"
        "  gcm_file: %s\n"
        "  target_grid_file: %s\n"
        "  input_channels: [tasmax_gcm, hurs_gcm, pr_gcm]\n"
        "  target_channels: [tasmax_era5, hurs_era5, pr_era5]\n"
        "model:\n"
        "  class_path: truss_downscaling.models.residual_unet.ResidualUNet\n"
        "  parameters: {in_channels: 3, out_channels: 3, depth: 1, base_width: 2}\n"
        "qdm:\n"
        "  variables: [pr]\n"
        "  input_file: %s\n"
        "  fit_window: ['2015-01-01', '2015-03-01']\n"
        "  wet_threshold_mm: 1.0\n"
        "  tail_clamp: true\n"
        "  time_chunk_size: 16\n"
        "%s"
        % (tmp_path / "runs", coarse_file, truth_file, model_file, extra_qdm),
        encoding="utf-8",
    )
    return config_path


def test_run_finalize_corrects_core_and_leaves_buffer_untouched(tmp_path):
    rng = np.random.default_rng(7)
    days, fit_days = 80, 60
    lat = np.arange(9.0)
    lon = np.arange(11.0)
    dates = np.arange("2015-01-01", "2015-03-22", dtype="datetime64[D]").astype("datetime64[ns]")
    shape = (days, lat.size, lon.size)

    truth = rng.gamma(2.0, 3.0, size=shape).astype(np.float32)
    truth[rng.random(truth.shape) < 0.2] = 0.0
    model = 0.64 * truth - 1.4
    model[model < WET_MM] = 0.0

    model_file = tmp_path / "model.nc"
    truth_file = tmp_path / "truth.nc"
    coarse_file = tmp_path / "coarse.nc"
    _write_synthetic_netcdf(model_file, model, lat, lon, dates)
    _write_synthetic_netcdf(truth_file, truth, lat, lon, dates)
    _write_synthetic_netcdf(coarse_file, np.zeros((2, 2, 2), dtype=np.float32), np.array([1.0, 6.0]), np.array([2.0, 8.0]), np.arange("2015-01-01", "2015-01-03", dtype="datetime64[D]").astype("datetime64[ns]"))

    config = load_config(_smoke_qdm_config(tmp_path, model_file, truth_file, coarse_file))
    destination = run_finalize(config)

    assert destination.exists()
    version_dir = destination.parent
    assert (version_dir / "qdm_fit.npz").exists()
    fit_manifest = json.loads((version_dir / "qdm_fit_manifest.json").read_text(encoding="utf-8"))
    assert fit_manifest["n_fit_days"] == fit_days
    assert len(fit_manifest["artifact_sha256"]) == 64
    stage_manifest = json.loads((version_dir / "manifest.json").read_text(encoding="utf-8"))
    assert stage_manifest["occurrence_and_tail_stats"]["pr"]["wet_invented"] > 0

    mask = core_mask(lat, lon, np.array([1.0, 6.0]), np.array([2.0, 8.0]))
    with xr.open_dataset(destination) as finalized, xr.open_dataset(model_file) as raw:
        corrected = finalized["pr"].values
        assert np.array_equal(corrected[:, ~mask], raw["pr"].values[:, ~mask])

        eval_slice = slice(fit_days, None)
        before = float(np.abs(model[eval_slice][:, mask] - truth[eval_slice][:, mask]).mean())
        after = float(np.abs(corrected[eval_slice][:, mask] - truth[eval_slice][:, mask]).mean())
        assert after < 0.55 * before
        wet_before = float((model[eval_slice][:, mask] >= WET_MM).mean())
        wet_after = float((corrected[eval_slice][:, mask] >= WET_MM).mean())
        wet_truth = float((truth[eval_slice][:, mask] >= WET_MM).mean())
        assert abs(wet_after - wet_truth) < abs(wet_before - wet_truth)


def test_run_chain_calls_finalize_only_when_enabled(tmp_path, monkeypatch, capsys):
    calls = []

    def _record(name):
        def stage(config, *args, **kwargs):
            calls.append(name)
            return tmp_path / f"{name}.nc"
        return stage

    for name in ("run_preprocess", "run_train", "run_evaluate", "run_infer", "run_finalize"):
        monkeypatch.setattr(cli, name, _record(name))

    config_path = _smoke_qdm_config(tmp_path, tmp_path / "model.nc", tmp_path / "truth.nc", tmp_path / "coarse.nc", extra_qdm="  enabled: true\n")
    monkeypatch.setattr(sys, "argv", ["truss-downscale", "run", "--config", str(config_path)])
    cli.main()
    assert calls == ["run_preprocess", "run_train", "run_evaluate", "run_infer", "run_finalize"]

    calls.clear()
    config_path = _smoke_qdm_config(tmp_path, tmp_path / "model.nc", tmp_path / "truth.nc", tmp_path / "coarse.nc", extra_qdm="  enabled: false\n")
    monkeypatch.setattr(sys, "argv", ["truss-downscale", "run", "--config", str(config_path)])
    cli.main()
    output = capsys.readouterr().out
    assert calls == ["run_preprocess", "run_train", "run_evaluate", "run_infer"]
    assert "finalize skipped" in output


def test_finalize_stage_invokes_run_finalize(tmp_path, monkeypatch):
    captured = {}

    def finalize(config, force):
        captured["force"] = force
        return tmp_path / "finalized.nc"

    monkeypatch.setattr(cli, "run_finalize", finalize)
    config_path = _smoke_qdm_config(tmp_path, tmp_path / "model.nc", tmp_path / "truth.nc", tmp_path / "coarse.nc")
    monkeypatch.setattr(sys, "argv", ["truss-downscale", "finalize", "--config", str(config_path), "--force"])
    cli.main()

    assert captured["force"] is True
