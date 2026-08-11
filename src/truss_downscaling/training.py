from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from .checkpointing import atomic_torch_save, mark_success, stage_is_complete
from .config import Config


class GlobalSSIML1(nn.Module):
    def __init__(self, alpha: float = 0.84) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1 = torch.abs(prediction - target).mean()
        mu_x, mu_y = prediction.mean(dim=(-2, -1), keepdim=True), target.mean(dim=(-2, -1), keepdim=True)
        var_x = ((prediction - mu_x) ** 2).mean(dim=(-2, -1), keepdim=True)
        var_y = ((target - mu_y) ** 2).mean(dim=(-2, -1), keepdim=True)
        cov = ((prediction - mu_x) * (target - mu_y)).mean(dim=(-2, -1), keepdim=True)
        c1, c2 = 0.01**2, 0.03**2
        ssim = ((2 * mu_x * mu_y + c1) * (2 * cov + c2) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))).mean()
        return self.alpha * (1 - ssim) + (1 - self.alpha) * l1


class ArrayDataset(Dataset):
    def __init__(self, stage: Path, indices: np.ndarray) -> None:
        self.x = np.load(stage / "X.npy", mmap_mode="r")
        self.y = np.load(stage / "Y.npy", mmap_mode="r")
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = int(self.indices[position])
        return torch.from_numpy(self.x[index].copy()), torch.from_numpy(self.y[index].copy())


def _loader(stage: Path, split: str, batch_size: int, shuffle: bool = False) -> DataLoader:
    with np.load(stage / "splits.npz", allow_pickle=False) as arrays:
        indices = arrays[split].copy()
    return DataLoader(ArrayDataset(stage, indices), batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=torch.cuda.is_available())


def _model(config: Config, device: torch.device) -> nn.Module:
    return config.model_class()(**config.model.get("parameters", {})).to(device)


def run_train(config: Config, force: bool = False, restart: bool = False) -> Path:
    stage = config.output_root / "preprocessing"
    if not stage_is_complete(stage):
        raise RuntimeError(f"preprocessing is incomplete: run preprocess first ({stage})")
    out = config.output_root / "training"
    out.mkdir(parents=True, exist_ok=True)
    if stage_is_complete(out) and not force:
        return out
    seed = int(config.run.get("seed", 42))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    requested_device = config.training.get("device", "auto")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if requested_device == "auto" else requested_device)
    model = _model(config, device)
    optimizer = AdamW(model.parameters(), lr=float(config.training.get("learning_rate", 1e-3)), weight_decay=float(config.training.get("weight_decay", 1e-4)))
    epochs = int(config.training.get("epochs", 30))
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = GlobalSSIML1(float(config.training.get("ssim_alpha", 0.84)))
    norm_stats = json.loads((stage / "normalization.json").read_text(encoding="utf-8"))
    start_epoch, best = 0, float("inf")
    last = out / "last.pt"
    if last.exists() and not restart:
        saved = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(saved["model_state"]); optimizer.load_state_dict(saved["optimizer_state"]); scheduler.load_state_dict(saved["scheduler_state"])
        start_epoch, best = saved["epoch"] + 1, saved["best_loss"]
    with np.load(stage / "splits.npz", allow_pickle=False) as splits:
        split_names = set(splits.files)
    training_split = "train" if "train" in split_names else "production"
    train_loader = _loader(stage, training_split, int(config.training.get("batch_size", 16)), shuffle=True)
    val_loader = _loader(stage, "validation", int(config.training.get("batch_size", 16))) if "validation" in split_names else None
    history = []
    for epoch in range(start_epoch, epochs):
        model.train(); train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device); optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb); loss.backward(); optimizer.step(); train_loss += loss.item() * len(xb)
        train_loss /= len(train_loader.dataset)
        model.eval(); val_loss = train_loss
        if val_loader:
            with torch.no_grad():
                val_loss = sum(criterion(model(xb.to(device)), yb.to(device)).item() * len(xb) for xb, yb in val_loader) / len(val_loader.dataset)
        scheduler.step()
        payload = {"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(), "best_loss": min(best, val_loss), "config": config.raw, "model_class": config.model.get("class_path"), "norm_stats": norm_stats}
        atomic_torch_save(payload, last)
        if val_loss < best:
            best = val_loss; atomic_torch_save(payload, out / "best.pt")
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss})
    with (out / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys() if history else ["epoch", "train_loss", "validation_loss"]); writer.writeheader(); writer.writerows(history)
    mark_success(out, {"best_loss": best, "epochs": epochs, "device": str(device), "model_class": config.model.get("class_path")})
    return out
