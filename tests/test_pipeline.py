import json
from pathlib import Path

import numpy as np
import torch

from truss_downscaling.config import load_config
from truss_downscaling.evaluation import run_evaluate
from truss_downscaling import preprocessing
from truss_downscaling.preprocessing import run_preprocess
from truss_downscaling.models.residual_unet import ResidualUNet
from truss_downscaling.training import run_train
from truss_downscaling.transforms import to_physical_targets, standardize_inputs


def test_model_preserves_era5_grid_shape():
    model = ResidualUNet(in_channels=3, out_channels=3, depth=2, base_width=4)
    result = model(torch.randn(1, 3, 181, 201))
    assert result.shape == (1, 3, 181, 201)


def test_config_imports_python_model():
    config = load_config(Path(__file__).parents[1] / "configs/development.example.yaml")
    assert config.model_class() is ResidualUNet


def test_training_requires_preprocessing(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "run: {id: smoke, output_root: %s}\n"
        "data: {input_channels: [a, b, c], target_channels: [x, y, z]}\n"
        "model: {class_path: truss_downscaling.models.residual_unet.ResidualUNet}\n"
        % tmp_path,
        encoding="utf-8",
    )
    config = load_config(config_path)
    try:
        run_train(config)
    except RuntimeError as error:
        assert "preprocessing is incomplete" in str(error)
    else:
        raise AssertionError("training should not run without preprocessing artifacts")


def _smoke_config(tmp_path: Path, *, epochs: int = 1) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "run:\n"
        "  id: smoke\n"
        "  output_root: %s\n"
        "  seed: 7\n"
        "data:\n"
        "  input_channels: [tasmax_gcm, hurs_gcm, pr_gcm]\n"
        "  target_channels: [tasmax_era5, hurs_era5, pr_era5]\n"
        "periods:\n"
        "  train: [2015-01-01, 2015-01-02]\n"
        "  validation: [2015-01-03, 2015-01-03]\n"
        "  test: [2015-01-04, 2015-01-04]\n"
        "preprocessing:\n"
        "  precipitation_transform: log1p\n"
        "model:\n"
        "  class_path: truss_downscaling.models.residual_unet.ResidualUNet\n"
        "  parameters: {in_channels: 3, out_channels: 3, depth: 1, base_width: 2}\n"
        "training:\n"
        "  epochs: %d\n"
        "  batch_size_per_device: 2\n"
        "  num_workers: 0\n"
        "  multi_gpu: false\n"
        "  device: cpu\n"
        "evaluation:\n"
        "  generate_plots: false\n"
        % (tmp_path / "runs", epochs),
        encoding="utf-8",
    )
    return config_path


def test_preprocessing_transforms_both_precipitation_arrays_and_uses_train_fill(tmp_path, monkeypatch):
    x = np.ones((4, 3, 2, 2), dtype=np.float32)
    y = np.ones((4, 3, 2, 2), dtype=np.float32)
    x[:, 2] = np.array([0, 1, 3, 7], dtype=np.float32)[:, None, None]
    y[:, 2] = np.array([0, 2, 4, 8], dtype=np.float32)[:, None, None]
    x[2, 0, 0, 0] = np.nan
    y[3, 1, 0, 0] = np.nan
    metadata = {
        "dates": ["2015-01-01", "2015-01-02", "2015-01-03", "2015-01-04"],
        "lat": [0, 1],
        "lon": [0, 1],
    }
    monkeypatch.setattr(preprocessing, "_load_pair", lambda config: (x.copy(), y.copy(), metadata.copy()))
    config = load_config(_smoke_config(tmp_path))

    run_preprocess(config)

    norm = json.loads(
        (config.output_root / "preprocessing" / "normalization.json").read_text(encoding="utf-8")
    )
    expected_x_mean = np.log1p(np.array([0, 1], dtype=np.float32)).mean()
    expected_y_mean = np.log1p(np.array([0, 2], dtype=np.float32)).mean()
    assert norm["normalization_version"] == 2
    assert norm["log1p_input_applied"] is True
    assert norm["log1p_target_applied"] is True
    assert np.isclose(norm["X_mean"][2], expected_x_mean)
    assert np.isclose(norm["Y_mean"][2], expected_y_mean)
    assert norm["nan_fill_policy"] == "training_split_channel_mean"

    normalized_x = np.load(config.output_root / "preprocessing" / "X.npy")
    normalized_y = np.load(config.output_root / "preprocessing" / "Y.npy")
    assert np.isfinite(normalized_x).all()
    assert np.isfinite(normalized_y).all()


def test_transform_helpers_invert_log1p_precipitation():
    norm = {
        "normalization_version": 2,
        "X_mean": [10, 50, np.log1p(2)],
        "X_std": [2, 5, 1],
        "Y_mean": [20, 60, np.log1p(4)],
        "Y_std": [2, 5, 1],
        "pr_index": 2,
        "target_pr_index": 2,
        "log1p_input_applied": True,
        "log1p_target_applied": True,
    }
    inputs = np.array([[[[10]], [[50]], [[2]]]], dtype=np.float32)
    standardized = standardize_inputs(inputs, norm)
    assert np.allclose(standardized, 0)
    target = np.array([[[[20]], [[60]], [[np.log1p(4)]]]], dtype=np.float32)
    physical = to_physical_targets((target - np.array(norm["Y_mean"])[None, :, None, None]) / np.array(norm["Y_std"])[None, :, None, None], norm)
    assert np.isclose(physical[0, 2, 0, 0], 4)


def test_training_and_evaluation_write_notebook_artifacts(tmp_path, monkeypatch):
    x = np.zeros((4, 3, 2, 2), dtype=np.float32)
    y = np.zeros((4, 3, 2, 2), dtype=np.float32)
    x[:, 0] = np.arange(4, dtype=np.float32)[:, None, None]
    y[:, 0] = x[:, 0] + 0.25
    x[:, 1] = 50
    y[:, 1] = 55
    x[:, 2] = np.arange(4, dtype=np.float32)[:, None, None]
    y[:, 2] = x[:, 2] + 1
    metadata = {
        "dates": ["2015-01-01", "2015-01-02", "2015-01-03", "2015-01-04"],
        "lat": [0, 1],
        "lon": [0, 1],
    }
    monkeypatch.setattr(preprocessing, "_load_pair", lambda config: (x.copy(), y.copy(), metadata.copy()))
    config = load_config(_smoke_config(tmp_path))
    run_preprocess(config)
    run_train(config)
    evaluation = run_evaluate(config)

    history = (config.output_root / "training" / "history.csv").read_text(encoding="utf-8")
    assert history.splitlines()[0] == "epoch,train_loss,val_loss,lr"
    checkpoint = torch.load(config.output_root / "training" / "best.pt", map_location="cpu", weights_only=False)
    assert checkpoint["checkpoint_version"] == 2
    assert checkpoint["epoch"] == 1
    assert (evaluation / "test_metrics.csv").exists()
    assert (evaluation / "bias_metrics.csv").exists()
    assert "rmse_phys" in (evaluation / "test_metrics.csv").read_text(encoding="utf-8").splitlines()[0]
