from pathlib import Path

import torch

from truss_downscaling.config import load_config
from truss_downscaling.models.residual_unet import ResidualUNet
from truss_downscaling.training import run_train


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
