"""Train entrypoint (baseline or fine-tune).

Examples:
  python train.py --stage baseline --epochs 15
  python train.py --stage finetune --epochs 15
"""
from __future__ import annotations

import argparse

from ra_effb3.config import Paths, make_data_config
from ra_effb3.training.baseline import train_baseline
from ra_effb3.training.finetune import train_finetune


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["baseline", "finetune"], required=True)
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()

    paths = Paths()
    cfg = make_data_config()

    if args.stage == "baseline":
        train_baseline(paths, cfg, epochs=args.epochs)
    else:
        train_finetune(paths, cfg, epochs=args.epochs)


if __name__ == "__main__":
    main()
