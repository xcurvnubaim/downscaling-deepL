from __future__ import annotations

import csv
import json
import logging
import platform
import random
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from .checkpointing import atomic_torch_save, mark_success, stage_is_complete
from .config import Config
from .transforms import validate_norm_stats


CHECKPOINT_VERSION = 2


class GlobalSSIML1(nn.Module):
    def __init__(self, alpha: float = 0.84) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1 = torch.abs(prediction - target).mean()
        mu_x = prediction.mean(dim=(-2, -1), keepdim=True)
        mu_y = target.mean(dim=(-2, -1), keepdim=True)
        var_x = ((prediction - mu_x) ** 2).mean(dim=(-2, -1), keepdim=True)
        var_y = ((target - mu_y) ** 2).mean(dim=(-2, -1), keepdim=True)
        cov = ((prediction - mu_x) * (target - mu_y)).mean(dim=(-2, -1), keepdim=True)
        c1, c2 = 0.01**2, 0.03**2
        ssim = (
            (2 * mu_x * mu_y + c1)
            * (2 * cov + c2)
            / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))
        ).mean()
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


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _loader(
    stage: Path,
    split: str,
    batch_size: int,
    shuffle: bool = False,
    *,
    seed: int = 42,
    num_workers: int = 2,
    pin_memory: bool | None = None,
) -> DataLoader:
    with np.load(stage / "splits.npz", allow_pickle=False) as arrays:
        if split not in arrays.files:
            raise RuntimeError(f"preprocessing artifacts do not contain a {split!r} split")
        indices = arrays[split].copy()
    generator = torch.Generator()
    generator.manual_seed(seed)
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return DataLoader(
        ArrayDataset(stage, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker if num_workers else None,
        generator=generator,
    )


def _resolve_device(config: Config) -> torch.device:
    requested = config.training.get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested device {requested!r}, but CUDA is unavailable")
    return device


def _use_data_parallel(config: Config, device: torch.device) -> bool:
    return (
        bool(config.training.get("multi_gpu", True))
        and device.type == "cuda"
        and torch.cuda.device_count() > 1
    )


def _batch_size(config: Config, device: torch.device) -> int:
    configured = config.training.get("batch_size_per_device")
    if configured is None:
        configured = config.training.get("batch_size", 16)
    per_device = int(configured)
    if per_device < 1:
        raise ValueError("training batch size must be positive")
    return per_device * torch.cuda.device_count() if _use_data_parallel(config, device) else per_device


def _model(config: Config, device: torch.device) -> nn.Module:
    return config.model_class()(**config.model.get("parameters", {})).to(device)


def _raw_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def _history_path(out: Path) -> Path:
    return out / "history.csv"


def _read_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return []
    if "val_loss" not in rows[0] or "lr" not in rows[0]:
        raise RuntimeError("training history uses an older format; rerun with --restart")
    return [
        {
            "epoch": int(row["epoch"]),
            "train_loss": float(row["train_loss"]),
            "val_loss": float(row["val_loss"]),
            "lr": float(row["lr"]),
        }
        for row in rows
    ]


def _write_history(path: Path, history: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = ["epoch", "train_loss", "val_loss", "lr"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def _execution_logger(out: Path) -> logging.Logger:
    logger = logging.getLogger(f"truss_downscaling.training.{out.resolve()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(out / "training.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def _maybe_start_wandb(
    config: Config,
    checkpoint: dict[str, Any] | None,
    *,
    device: torch.device,
    batch_size: int,
    train_samples: int,
    validation_samples: int,
) -> Any:
    tracking = config.raw.get("tracking", {})
    if not tracking.get("enabled", False):
        return None
    try:
        import wandb

        run_config = {
            "run_id": config.run.get("id", config.path.stem),
            "seed": int(config.run.get("seed", 42)),
            "device": str(device),
            "batch_size": batch_size,
            "train_samples": train_samples,
            "validation_samples": validation_samples,
            "epochs": int(config.training.get("epochs", 300)),
            "lr": float(config.training.get("learning_rate", 1e-3)),
            "weight_decay": float(config.training.get("weight_decay", 1e-4)),
            "ssim_alpha": float(config.training.get("ssim_alpha", 0.84)),
            **config.model.get("parameters", {}),
            "channels": list(config.data["input_channels"]),
        }
        kwargs = {
            "project": tracking.get("project", "truss-downscaling"),
            "name": tracking.get("run_name", config.run.get("id")),
            "config": run_config,
            "resume": "allow",
        }
        if checkpoint and checkpoint.get("wandb_run_id"):
            kwargs["id"] = checkpoint["wandb_run_id"]
        run = wandb.init(**kwargs)
        run.summary.update(
            {
                "execution/hostname": socket.gethostname(),
                "execution/python": platform.python_version(),
                "execution/pytorch": torch.__version__,
                "execution/platform": platform.platform(),
                "execution/cuda_available": torch.cuda.is_available(),
                "execution/gpu_count": torch.cuda.device_count(),
                "execution/command": " ".join(sys.argv),
                "execution/status": "running",
            }
        )
        return run
    except Exception as error:  # W&B is deliberately non-critical.
        print(f"wandb unavailable ({type(error).__name__}); logging to CSV only")
        return None


def _upload_wandb_artifact(config: Config, run: Any, out: Path, best_loss: float) -> None:
    artifact_config = config.tracking.get("artifacts", {})
    if not artifact_config.get("enabled", True):
        return
    try:
        import wandb

        artifact = wandb.Artifact(
            name=artifact_config.get("name", f"{config.run.get('id', config.path.stem)}-model"),
            type=artifact_config.get("type", "model"),
            description="Best validation checkpoint from a completed downscaling training run",
            metadata={
                "run_id": config.run.get("id", config.path.stem),
                "best_val_loss": best_loss,
                "checkpoint_version": CHECKPOINT_VERSION,
                "model_class": config.model.get("class_path"),
            },
        )
        artifact.add_file(str(out / "best.pt"), name="best.pt")
        artifact.add_file(str(out / "history.csv"), name="history.csv")
        artifact.add_file(str(out / "manifest.json"), name="manifest.json")
        run.log_artifact(artifact, aliases=list(artifact_config.get("aliases", ["latest", "best"])))
        run.summary["artifact/name"] = artifact.name
        run.summary["artifact/status"] = "uploaded"
    except Exception as error:  # Local checkpoints must remain usable if artifact storage is down.
        run.summary["artifact/status"] = "failed"
        run.summary["artifact/error"] = f"{type(error).__name__}: {error}"
        logging.getLogger(f"truss_downscaling.training.{out.resolve()}").exception(
            "W&B checkpoint artifact upload failed"
        )


def run_train(config: Config, force: bool = False, restart: bool = False) -> Path:
    stage = config.output_root / "preprocessing"
    if not stage_is_complete(stage):
        raise RuntimeError(f"preprocessing is incomplete: run preprocess first ({stage})")
    out = config.output_root / "training"
    out.mkdir(parents=True, exist_ok=True)
    if stage_is_complete(out) and not force:
        return out
    logger = _execution_logger(out)
    started_at = time.perf_counter()
    logger.info("Training started run_id=%s config=%s", config.run.get("id", config.path.stem), config.path)

    seed = int(config.run.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = _resolve_device(config)
    model = _model(config, device)
    use_data_parallel = _use_data_parallel(config, device)
    if use_data_parallel:
        model = nn.DataParallel(model)

    optimizer = AdamW(
        model.parameters(),
        lr=float(config.training.get("learning_rate", 1e-3)),
        weight_decay=float(config.training.get("weight_decay", 1e-4)),
    )
    epochs = int(config.training.get("epochs", 300))
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = GlobalSSIML1(float(config.training.get("ssim_alpha", 0.84)))
    norm_stats = json.loads((stage / "normalization.json").read_text(encoding="utf-8"))
    validate_norm_stats(norm_stats)

    start_epoch = 1
    best = float("inf")
    saved: dict[str, Any] | None = None
    last = out / "last.pt"
    history = [] if restart else _read_history(_history_path(out))
    if last.exists() and not restart:
        saved = torch.load(last, map_location=device, weights_only=False)
        if int(saved.get("checkpoint_version", 0)) != CHECKPOINT_VERSION:
            raise RuntimeError("training checkpoint uses an older format; rerun with --restart")
        _raw_model(model).load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        scheduler.load_state_dict(saved["scheduler_state"])
        start_epoch = int(saved["epoch"]) + 1
        best = float(saved["best_val_loss"])

    with np.load(stage / "splits.npz", allow_pickle=False) as splits:
        split_names = set(splits.files)
    training_split = "train" if "train" in split_names else "production"
    batch_size = _batch_size(config, device)
    num_workers = int(config.training.get("num_workers", 2))
    train_loader = _loader(
        stage,
        training_split,
        batch_size,
        shuffle=True,
        seed=seed,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = (
        _loader(
            stage,
            "validation",
            batch_size,
            seed=seed + 1,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
        if "validation" in split_names
        else None
    )

    wandb_run = _maybe_start_wandb(
        config,
        saved,
        device=device,
        batch_size=batch_size,
        train_samples=len(train_loader.dataset),
        validation_samples=len(val_loader.dataset) if val_loader is not None else 0,
    )
    logger.info(
        "Execution device=%s data_parallel=%s batch_size=%d train_samples=%d validation_samples=%d start_epoch=%d",
        device,
        use_data_parallel,
        batch_size,
        len(train_loader.dataset),
        len(val_loader.dataset) if val_loader is not None else 0,
        start_epoch,
    )
    log_every = max(1, int(config.tracking.get("log_every_n_steps", 10)))
    global_step = (start_epoch - 1) * len(train_loader)
    try:
        for epoch in range(start_epoch, epochs + 1):
            epoch_started_at = time.perf_counter()
            model.train()
            train_total = 0.0
            for batch_index, (xb, yb) in enumerate(train_loader, start=1):
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                batch_loss = loss.item()
                train_total += batch_loss * len(xb)
                global_step += 1
                if wandb_run is not None and (batch_index % log_every == 0 or batch_index == len(train_loader)):
                    wandb_run.log(
                        {
                            "train/batch_loss": batch_loss,
                            "train/epoch": epoch,
                            "train/batch": batch_index,
                        },
                        step=global_step,
                    )
            train_loss = train_total / len(train_loader.dataset)

            model.eval()
            val_loss = train_loss
            if val_loader is not None:
                val_total = 0.0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        val_total += criterion(model(xb.to(device)), yb.to(device)).item() * len(xb)
                val_loss = val_total / len(val_loader.dataset)

            scheduler.step()
            learning_rate = scheduler.get_last_lr()[0]
            epoch_seconds = time.perf_counter() - epoch_started_at
            is_best = val_loss < best
            if is_best:
                best = val_loss
            payload = {
                "checkpoint_version": CHECKPOINT_VERSION,
                "epoch": epoch,
                "model_state": _raw_model(model).state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_loss": best,
                "val_loss": val_loss,
                "config": config.raw,
                "model_class": config.model.get("class_path"),
                "norm_stats": norm_stats,
            }
            if wandb_run is not None:
                payload["wandb_run_id"] = wandb_run.id
            atomic_torch_save(payload, last)
            if is_best:
                atomic_torch_save(payload, out / "best.pt")

            row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": learning_rate}
            history = [item for item in history if int(item["epoch"]) != epoch]
            history.append(row)
            history.sort(key=lambda item: int(item["epoch"]))
            _write_history(_history_path(out), history)
            logger.info(
                "Epoch %d/%d train_loss=%.6f val_loss=%.6f lr=%.8g seconds=%.2f best=%s",
                epoch,
                epochs,
                train_loss,
                val_loss,
                learning_rate,
                epoch_seconds,
                is_best,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "lr": learning_rate,
                        "execution/epoch_seconds": epoch_seconds,
                        "execution/samples_per_second": len(train_loader.dataset) / epoch_seconds,
                    },
                    step=global_step,
                )

        _write_history(_history_path(out), history)
        duration = time.perf_counter() - started_at
        manifest = {
            "best_loss": best,
            "epochs": epochs,
            "device": str(device),
            "model_class": config.model.get("class_path"),
            "batch_size": batch_size,
            "data_parallel": use_data_parallel,
            "duration_seconds": duration,
        }
        mark_success(out, manifest)
        logger.info("Training completed best_val_loss=%.6f duration_seconds=%.2f", best, duration)
        if wandb_run is not None:
            wandb_run.summary.update(
                {
                    "best_val_loss": best,
                    "execution/duration_seconds": duration,
                    "execution/status": "completed",
                }
            )
            _upload_wandb_artifact(config, wandb_run, out, best)
    except BaseException as error:
        logger.exception("Training failed: %s", error)
        if wandb_run is not None:
            wandb_run.summary["execution/status"] = "failed"
            wandb_run.summary["execution/error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        for handler in logger.handlers:
            handler.close()
        logger.handlers.clear()
    return out
