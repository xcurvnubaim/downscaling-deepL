from __future__ import annotations

import argparse
from pathlib import Path

from .artifacts import download_wandb_checkpoint
from .config import load_config
from .evaluation import run_evaluate
from .inference import run_infer
from .preprocessing import run_preprocess
from .training import run_train


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the staged TRUSS UKESM1 to ERA5 pipeline")
    parser.add_argument("stage", choices=("preprocess", "train", "evaluate", "infer", "run"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--restart", action="store_true", help="start training from fresh weights")
    parser.add_argument("--input-file", help="override data.gcm_file for inference")
    parser.add_argument("--artifact", help="W&B model artifact: ENTITY/PROJECT/NAME:ALIAS")
    parser.add_argument(
        "--artifact-dir",
        default="artifacts",
        help="directory for downloaded W&B artifacts (default: artifacts)",
    )
    args = parser.parse_args()
    if args.stage != "infer" and (args.input_file or args.artifact):
        parser.error("--input-file and --artifact are only valid for the infer stage")
    config = load_config(args.config)
    if args.stage in ("preprocess", "run"):
        run_preprocess(config, force=args.force)
    if args.stage in ("train", "run"):
        run_train(config, force=args.force, restart=args.restart)
    if args.stage in ("evaluate", "run"):
        run_evaluate(config, force=args.force)
    if args.stage == "infer":
        if args.input_file:
            config.data["gcm_file"] = str(Path(args.input_file).resolve())
        if args.artifact:
            checkpoint = download_wandb_checkpoint(args.artifact, args.artifact_dir)
            config.data["checkpoint"] = str(checkpoint)
        run_infer(config, force=args.force)


if __name__ == "__main__":
    main()
