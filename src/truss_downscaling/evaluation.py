from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpointing import mark_success, stage_is_complete
from .config import Config
from .training import (
    _batch_size,
    _loader,
    _model,
    _raw_model,
    _resolve_device,
    _use_data_parallel,
)
from .transforms import (
    to_physical_inputs,
    to_physical_targets,
    validate_norm_stats,
)


def _global_ssim(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    mu_p = prediction.mean(axis=(-2, -1), keepdims=True)
    mu_t = target.mean(axis=(-2, -1), keepdims=True)
    var_p = ((prediction - mu_p) ** 2).mean(axis=(-2, -1), keepdims=True)
    var_t = ((target - mu_t) ** 2).mean(axis=(-2, -1), keepdims=True)
    cov = ((prediction - mu_p) * (target - mu_t)).mean(axis=(-2, -1), keepdims=True)
    c1, c2 = 0.01**2, 0.03**2
    return (
        (2 * mu_p * mu_t + c1)
        * (2 * cov + c2)
        / ((mu_p**2 + mu_t**2 + c1) * (var_p + var_t + c2))
    ).reshape(prediction.shape[0])


def _bias_summary(
    sums: np.ndarray,
    squares: np.ndarray,
    absolute: np.ndarray,
    count: int,
) -> list[dict[str, float]]:
    mean = sums / count
    std = np.sqrt(np.maximum(squares / count - mean**2, 0))
    return [
        {
            "bias_mean": float(mean[index]),
            "bias_std": float(std[index]),
            "mae": float(absolute[index] / count),
            "rmse": float(np.sqrt(squares[index] / count)),
        }
        for index in range(len(sums))
    ]


def _write_map(
    output: Path,
    channel_name: str,
    date: str,
    lat: np.ndarray,
    lon: np.ndarray,
    input_field: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
) -> Path:
    try:
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError as error:
        raise RuntimeError(
            "diagnostic maps require the plots extra; install truss-downscaling[plots]"
        ) from error

    bias_initial = input_field - target
    bias_post = prediction - target
    improvement = np.abs(bias_initial) - np.abs(bias_post)
    units = {"tasmax": "degC", "hurs": "%", "pr": "mm day-1"}.get(channel_name, "")
    lon2d, lat2d = np.meshgrid(lon, lat)
    value_min = min(input_field.min(), prediction.min(), target.min())
    value_max = max(input_field.max(), prediction.max(), target.max())
    bias_max = max(np.abs(bias_initial).max(), np.abs(bias_post).max())
    improvement_max = np.abs(improvement).max()

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(17, 12),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    specs = [
        (axes[0, 0], f"GCM input ({channel_name}_gcm)", input_field, "viridis", value_min, value_max),
        (axes[0, 1], f"Prediction (Residual U-Net {channel_name})", prediction, "viridis", value_min, value_max),
        (axes[0, 2], f"Target / ERA5 ({channel_name}_era5)", target, "viridis", value_min, value_max),
        (axes[1, 0], f"Initial bias (GCM - target) [{units}]", bias_initial, "RdBu_r", -bias_max, bias_max),
        (axes[1, 1], f"Post-prediction bias (pred - target) [{units}]", bias_post, "RdBu_r", -bias_max, bias_max),
        (axes[1, 2], f"Bias reduction |init| - |post| [{units}]", improvement, "BrBG", -improvement_max, improvement_max),
    ]
    images = []
    for axis, title, field, cmap, lo, hi in specs:
        axis.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
        axis.add_feature(cfeature.COASTLINE, linewidth=0.5)
        axis.set_title(title, fontsize=11)
        gridlines = axis.gridlines(
            draw_labels=True,
            dms=True,
            x_inline=False,
            y_inline=False,
            color="k",
            alpha=0.3,
            linewidth=0.4,
        )
        gridlines.top_labels = False
        gridlines.right_labels = False
        images.append(
            axis.pcolormesh(
                lon2d,
                lat2d,
                field,
                cmap=cmap,
                vmin=lo,
                vmax=hi,
                shading="auto",
                transform=ccrs.PlateCarree(),
            )
        )
    fig.colorbar(images[0], ax=axes[0].tolist(), orientation="horizontal", shrink=0.7, pad=0.06, label=units)
    fig.colorbar(images[3], ax=axes[1, :2].tolist(), orientation="horizontal", shrink=0.7, pad=0.06, label=units)
    fig.colorbar(images[5], ax=axes[1, 2], orientation="horizontal", shrink=0.9, pad=0.06, label=units)
    initial_rmse = float(np.sqrt(np.mean(bias_initial**2)))
    post_rmse = float(np.sqrt(np.mean(bias_post**2)))
    change = 100 * (initial_rmse - post_rmse) / initial_rmse if initial_rmse else 0.0
    fig.suptitle(
        f"{date} | {channel_name} | bias RMSE {initial_rmse:.2f} -> {post_rmse:.2f} "
        f"{units} ({change:+.1f}%)",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout()
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / f"{channel_name}_map_input_pred_target_bias.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _log_metrics_to_wandb(config: Config, metrics: list[dict[str, Any]]) -> None:
    if not config.tracking.get("enabled", False):
        return
    try:
        import wandb

        run = wandb.init(
            project=config.tracking.get("project", "truss-downscaling"),
            name=f"{config.tracking.get('run_name', config.run.get('id', 'run'))}-evaluation",
            job_type="evaluation",
        )
        for metric in metrics:
            channel = metric["channel"]
            run.log(
                {
                    f"test/{channel}_rmse": metric["rmse"],
                    f"test/{channel}_mae": metric["mae"],
                    f"test/{channel}_ssim": metric["ssim"],
                    f"test/{channel}_rmse_phys": metric["rmse_phys"],
                }
            )
        run.finish()
    except Exception as error:  # W&B is deliberately non-critical.
        print(f"wandb evaluation logging unavailable ({type(error).__name__}); CSV retained")


def run_evaluate(config: Config, force: bool = False) -> Path:
    preprocessing = config.output_root / "preprocessing"
    training = config.output_root / "training"
    if not stage_is_complete(preprocessing) or not (training / "best.pt").exists():
        raise RuntimeError("evaluation requires completed preprocessing and training stages")
    with np.load(preprocessing / "splits.npz", allow_pickle=False) as splits:
        if "test" not in splits.files:
            raise RuntimeError("evaluation requires a test split; production training has no test evaluation")
        test_indices = splits["test"].copy()
    output = config.output_root / "evaluation"
    if stage_is_complete(output) and not force:
        return output
    output.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(config)
    model = _model(config, device)
    if _use_data_parallel(config, device):
        model = torch.nn.DataParallel(model)
    checkpoint = torch.load(training / "best.pt", map_location=device, weights_only=False)
    if int(checkpoint.get("checkpoint_version", 0)) != 2:
        raise RuntimeError("training checkpoint uses an older format; retrain with --restart")
    norm = checkpoint.get("norm_stats")
    if norm is None:
        norm = json.loads((preprocessing / "normalization.json").read_text(encoding="utf-8"))
    validate_norm_stats(norm)
    _raw_model(model).load_state_dict(checkpoint["model_state"])
    model.eval()

    batch_size = _batch_size(config, device)
    loader = _loader(
        preprocessing,
        "test",
        batch_size,
        num_workers=int(config.training.get("num_workers", 2)),
        pin_memory=device.type == "cuda",
    )
    channels = list(config.data["target_channels"])
    channel_count = len(channels)
    squared = np.zeros(channel_count, dtype=np.float64)
    absolute = np.zeros(channel_count, dtype=np.float64)
    ssim_total = np.zeros(channel_count, dtype=np.float64)
    physical_squared = np.zeros(channel_count, dtype=np.float64)
    before_sums = np.zeros(channel_count, dtype=np.float64)
    before_squares = np.zeros(channel_count, dtype=np.float64)
    before_absolute = np.zeros(channel_count, dtype=np.float64)
    after_sums = np.zeros(channel_count, dtype=np.float64)
    after_squares = np.zeros(channel_count, dtype=np.float64)
    after_absolute = np.zeros(channel_count, dtype=np.float64)
    pixel_count = 0
    sample_count = 0
    map_channel = config.evaluation.get("map_channel", "tasmax_era5")
    if map_channel not in channels:
        raise ValueError(f"evaluation.map_channel must be one of {channels}, got {map_channel!r}")
    map_index = channels.index(map_channel)
    best_target_value = -np.inf
    map_fields: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    map_global_index: int | None = None
    offset = 0

    with torch.no_grad():
        for inputs, targets in loader:
            predictions = model(inputs.to(device, non_blocking=True)).cpu().numpy()
            input_values = inputs.numpy()
            target_values = targets.numpy()
            difference = predictions - target_values
            batch_size_actual = len(inputs)
            pixel_count += batch_size_actual * predictions.shape[-2] * predictions.shape[-1]
            sample_count += batch_size_actual
            squared += (difference**2).sum(axis=(0, 2, 3))
            absolute += np.abs(difference).sum(axis=(0, 2, 3))
            for channel in range(channel_count):
                ssim_total[channel] += _global_ssim(predictions[:, channel], target_values[:, channel]).sum()

            inputs_physical = to_physical_inputs(input_values, norm)
            predictions_physical = to_physical_targets(predictions, norm)
            targets_physical = to_physical_targets(target_values, norm)
            physical_difference = predictions_physical - targets_physical
            physical_squared += (physical_difference**2).sum(axis=(0, 2, 3))
            initial_bias = inputs_physical - targets_physical
            post_bias = predictions_physical - targets_physical
            before_sums += initial_bias.sum(axis=(0, 2, 3))
            before_squares += (initial_bias**2).sum(axis=(0, 2, 3))
            before_absolute += np.abs(initial_bias).sum(axis=(0, 2, 3))
            after_sums += post_bias.sum(axis=(0, 2, 3))
            after_squares += (post_bias**2).sum(axis=(0, 2, 3))
            after_absolute += np.abs(post_bias).sum(axis=(0, 2, 3))

            hottest = targets_physical[:, map_index].max(axis=(1, 2))
            local_index = int(np.argmax(hottest))
            if hottest[local_index] > best_target_value:
                best_target_value = float(hottest[local_index])
                map_fields = (
                    inputs_physical[local_index, map_index].copy(),
                    predictions_physical[local_index, map_index].copy(),
                    targets_physical[local_index, map_index].copy(),
                )
                map_global_index = int(test_indices[offset + local_index])
            offset += batch_size_actual

    metrics = []
    for channel in range(channel_count):
        metrics.append(
            {
                "split": "test",
                "channel": channels[channel],
                "rmse": float(np.sqrt(squared[channel] / pixel_count)),
                "mae": float(absolute[channel] / pixel_count),
                "ssim": float(ssim_total[channel] / sample_count),
                "rmse_phys": float(np.sqrt(physical_squared[channel] / pixel_count)),
            }
        )
    with (output / "test_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["split", "channel", "rmse", "mae", "ssim", "rmse_phys"])
        writer.writeheader()
        writer.writerows(metrics)

    before = _bias_summary(before_sums, before_squares, before_absolute, pixel_count)
    after = _bias_summary(after_sums, after_squares, after_absolute, pixel_count)
    with (output / "bias_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = ["phase", "channel", "bias_mean", "bias_std", "mae", "rmse", "rmse_reduction_pct"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for phase, rows in (("before", before), ("after", after)):
            for channel, row in zip(channels, rows):
                reduction = ""
                if phase == "after" and before[channels.index(channel)]["rmse"]:
                    reduction = 100 * (before[channels.index(channel)]["rmse"] - row["rmse"]) / before[channels.index(channel)]["rmse"]
                writer.writerow({"phase": phase, "channel": channel, **row, "rmse_reduction_pct": reduction})

    figure_path = None
    if config.evaluation.get("generate_plots", False):
        if map_fields is None or map_global_index is None:
            raise RuntimeError("cannot generate a diagnostic map without test samples")
        metadata = json.loads((preprocessing / "metadata.json").read_text(encoding="utf-8"))
        dates = metadata["dates"]
        figure_path = _write_map(
            output,
            map_channel.removesuffix("_era5"),
            dates[map_global_index],
            np.asarray(metadata["lat"]),
            np.asarray(metadata["lon"]),
            *map_fields,
        )

    _log_metrics_to_wandb(config, metrics)
    mark_success(
        output,
        {
            "checkpoint": str(training / "best.pt"),
            "channels": channels,
            "test_metrics": metrics,
            "bias_metrics": str(output / "bias_metrics.csv"),
            "figure": str(figure_path) if figure_path else None,
        },
    )
    return output
