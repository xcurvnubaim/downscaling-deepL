from __future__ import annotations

import argparse

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
    args = parser.parse_args()
    config = load_config(args.config)
    if args.stage in ("preprocess", "run"):
        run_preprocess(config, force=args.force)
    if args.stage in ("train", "run"):
        run_train(config, force=args.force, restart=args.restart)
    if args.stage in ("evaluate", "run"):
        run_evaluate(config, force=args.force)
    if args.stage == "infer":
        run_infer(config, force=args.force)


if __name__ == "__main__":
    main()
