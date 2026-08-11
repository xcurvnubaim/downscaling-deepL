from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from .checkpointing import mark_success, stage_is_complete
from .config import Config
from .training import _loader, _model


def run_evaluate(config: Config, force: bool = False) -> Path:
    preprocessing = config.output_root / "preprocessing"
    training = config.output_root / "training"
    if not stage_is_complete(preprocessing) or not (training / "best.pt").exists():
        raise RuntimeError("evaluation requires completed preprocessing and training stages")
    if not (preprocessing / "splits.npz").exists() or "test" not in np.load(preprocessing / "splits.npz"):
        raise RuntimeError("evaluation requires a test split; production training has no test evaluation")
    output = config.output_root / "evaluation"
    if stage_is_complete(output) and not force:
        return output
    output.mkdir(parents=True, exist_ok=True)
    requested = config.training.get("device", "auto")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested)
    model = _model(config, device)
    checkpoint = torch.load(training / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    loader = _loader(preprocessing, "test", int(config.training.get("batch_size", 16)))
    errors = []
    with torch.no_grad():
        for inputs, targets in loader:
            errors.append((model(inputs.to(device)).cpu().numpy() - targets.numpy()) ** 2)
    rmse = np.sqrt(np.concatenate(errors, axis=0).mean(axis=(0, 2, 3)))
    with (output / "test_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(["channel", "standardized_rmse"])
        writer.writerows(zip(config.data["target_channels"], rmse.tolist()))
    mark_success(output, {"checkpoint": str(training / "best.pt"), "channels": config.data["target_channels"], "standardized_rmse": rmse.tolist()})
    return output
