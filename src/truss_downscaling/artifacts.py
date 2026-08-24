from __future__ import annotations

from pathlib import Path


def download_wandb_checkpoint(reference: str, output_root: str | Path = "artifacts") -> Path:
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "downloading a W&B artifact requires the tracking extra: pip install -e '.[tracking]'"
        ) from error

    destination = Path(output_root).resolve() / reference.rsplit("/", 1)[-1].replace(":", "-")
    artifact = wandb.Api().artifact(reference, type="model")
    artifact_dir = Path(artifact.download(root=str(destination))).resolve()
    checkpoint = artifact_dir / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"artifact downloaded to {artifact_dir}, but it does not contain best.pt"
        )
    return checkpoint
