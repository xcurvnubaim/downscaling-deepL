#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import wandb


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a W&B model artifact for downscaling inference."
    )
    parser.add_argument(
        "artifact",
        help="Artifact reference: ENTITY/PROJECT/NAME:ALIAS",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts"),
        help="Download directory (default: artifacts)",
    )
    args = parser.parse_args()

    artifact = wandb.Api().artifact(args.artifact, type="model")
    artifact_dir = Path(artifact.download(root=str(args.output))).resolve()
    checkpoint = artifact_dir / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"artifact downloaded to {artifact_dir}, but it does not contain best.pt"
        )

    print(checkpoint)


if __name__ == "__main__":
    main()
