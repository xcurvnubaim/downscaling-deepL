#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from truss_downscaling.artifacts import download_wandb_checkpoint


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

    checkpoint = download_wandb_checkpoint(args.artifact, args.output)
    print(checkpoint)


if __name__ == "__main__":
    main()
