import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from truss_downscaling import inference, preprocessing
from truss_downscaling.config import load_config
from truss_downscaling.preprocessing import run_preprocess
from truss_downscaling.regridding import (
    describe_regridding,
    regrid_methods_for,
    resolve_regrid_methods,
)

BASE = (
    "run:\n"
    "  id: regrid\n"
    "  output_root: %s\n"
    "data:\n"
    "  input_channels: [tasmax_gcm, hurs_gcm, pr_gcm]\n"
    "  target_channels: [tasmax_era5, hurs_era5, pr_era5]\n"
    "periods:\n"
    "  train: [2015-01-01, 2015-01-02]\n"
    "model:\n"
    "  class_path: truss_downscaling.models.residual_unet.ResidualUNet\n"
    "  parameters: {in_channels: 3, out_channels: 3, depth: 1, base_width: 2}\n"
)


def _config(tmp_path, preprocessing_block: str = ""):
    text = BASE % (tmp_path / "runs")
    if preprocessing_block:
        text += "preprocessing:\n" + preprocessing_block
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return load_config(path)


def test_regridding_defaults_to_bilinear(tmp_path):
    methods = resolve_regrid_methods(_config(tmp_path))
    assert methods == {"tasmax": "bilinear", "hurs": "bilinear", "pr": "bilinear"}


def test_regridding_can_be_set_globally(tmp_path):
    methods = resolve_regrid_methods(_config(tmp_path, "  regridding: conservative\n"))
    assert set(methods.values()) == {"conservative"}


def test_regridding_can_be_set_per_variable(tmp_path):
    block = "  regridding: bilinear\n  regridding_by_variable:\n    pr: conservative\n"
    methods = resolve_regrid_methods(_config(tmp_path, block))
    assert methods == {"tasmax": "bilinear", "hurs": "bilinear", "pr": "conservative"}


def test_regridding_rejects_unknown_method(tmp_path):
    with pytest.raises(ValueError, match="unsupported regridding method"):
        resolve_regrid_methods(_config(tmp_path, "  regridding: cubic\n"))


def test_regridding_rejects_unknown_variable(tmp_path):
    block = "  regridding_by_variable:\n    pr_gcm: conservative\n"
    with pytest.raises(ValueError, match="not input channels"):
        resolve_regrid_methods(_config(tmp_path, block))


def test_regridding_description_marks_smoothing_per_variable():
    uniform = describe_regridding({"tasmax": "bilinear", "hurs": "bilinear", "pr": "bilinear"})
    assert uniform["label"] == "xesmf_bilinear"
    assert uniform["uniform"] is True
    assert all(uniform["smoothing"].values())

    mixed = describe_regridding({"tasmax": "bilinear", "hurs": "bilinear", "pr": "conservative"})
    assert mixed["label"] == "xesmf_mixed"
    assert mixed["uniform"] is False
    assert mixed["by_variable"]["pr"] == "conservative"
    assert mixed["smoothing"]["pr"] is False
    assert mixed["smoothing"]["tasmax"] is True


def test_serving_uses_the_method_recorded_at_preprocessing(tmp_path):
    norm = {"regrid_methods": {"tasmax": "bilinear", "hurs": "bilinear", "pr": "conservative"}}
    config = _config(tmp_path)
    assert regrid_methods_for(norm, config)["pr"] == "conservative"


def test_serving_rejects_a_config_that_contradicts_the_artifact(tmp_path):
    norm = {"regrid_methods": {"tasmax": "bilinear", "hurs": "bilinear", "pr": "conservative"}}
    config = _config(tmp_path, "  regridding: bilinear\n")
    with pytest.raises(ValueError, match="disagrees with the preprocessing"):
        regrid_methods_for(norm, config)


def test_serving_accepts_a_config_that_matches_the_artifact(tmp_path):
    norm = {"regrid_methods": {"tasmax": "bilinear", "hurs": "bilinear", "pr": "conservative"}}
    block = "  regridding_by_variable:\n    pr: conservative\n"
    assert regrid_methods_for(norm, _config(tmp_path, block))["pr"] == "conservative"


def test_serving_falls_back_to_config_for_old_artifacts(tmp_path):
    config = _config(tmp_path, "  regridding_by_variable:\n    pr: nearest_s2d\n")
    assert regrid_methods_for({}, config)["pr"] == "nearest_s2d"


def test_preprocessing_records_the_methods(tmp_path, monkeypatch):
    x = np.ones((4, 3, 2, 2), dtype=np.float32)
    y = np.ones((4, 3, 2, 2), dtype=np.float32)
    methods = {"tasmax": "bilinear", "hurs": "bilinear", "pr": "conservative"}
    metadata = {
        "dates": ["2015-01-01", "2015-01-02", "2015-01-03", "2015-01-04"],
        "lat": [0, 1],
        "lon": [0, 1],
        "regrid_methods": methods,
    }
    monkeypatch.setattr(preprocessing, "_load_pair", lambda config: (x.copy(), y.copy(), metadata.copy()))
    config = _config(tmp_path, "  regridding_by_variable:\n    pr: conservative\n")

    run_preprocess(config)

    norm = json.loads(
        (config.output_root / "preprocessing" / "normalization.json").read_text(encoding="utf-8")
    )
    assert norm["regrid_methods"] == methods


class _FakeValue:
    def __init__(self, fill: float):
        self.values = np.full((2, 3, 4), fill, dtype="float32")

    def isel(self, indexers):
        return self


class _FakeSource:
    def __init__(self, fill: float = 0.0):
        self.fill = fill
        self.requested: list[object] = []

    def __getitem__(self, key):
        self.requested.append(key)
        if isinstance(key, list):
            return self
        return _FakeValue(self.fill)

    def load(self):
        return self


class _FakeRegridder:
    scale = {"bilinear": 1.0, "conservative": 2.0, "nearest_s2d": 3.0}

    def __init__(self, source, target, method="bilinear", periodic=False):
        self.method = method

    def __call__(self, values):
        return _FakeValue(self.scale[self.method])


def test_inference_worker_routes_each_variable_to_its_own_method(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "xesmf", SimpleNamespace(Regridder=_FakeRegridder))

    def fake_open_dataset(path, *args, **kwargs):
        return _FakeSource()

    monkeypatch.setattr(inference.xr, "open_dataset", fake_open_dataset)

    inference._init_regrid_worker(
        "source.nc",
        "target.nc",
        ["tasmax", "hurs", "pr"],
        False,
        {"tasmax": "bilinear", "hurs": "bilinear", "pr": "conservative"},
    )

    assert sorted(inference._WORKER_REGRIDDERS) == ["bilinear", "conservative"]
    assert inference._regrid_channel(("tasmax", "time", 0, 2)).max() == pytest.approx(1.0)
    assert inference._regrid_channel(("pr", "time", 0, 2)).max() == pytest.approx(2.0)
