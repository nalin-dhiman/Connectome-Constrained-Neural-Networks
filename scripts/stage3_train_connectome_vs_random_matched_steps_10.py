#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

import stage3_train_connectome_vs_random_matched_steps as base


DEFAULT_OUTPUT_ROOT = Path("results/main_results/stage3_connectome_vs_random_matched_steps_10")
DEFAULT_REPORT_PATH = Path(
    "docs/generated/stage3_connectome_vs_random_matched_steps_10_seed_summary.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Matched-step Stage 3 connectome vs random-mask baseline comparison (10 iterations)."
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-iters", type=int, default=10)
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=base.BASELINE_CHECKPOINT,
    )
    parser.add_argument(
        "--random-mask-path",
        type=Path,
        default=base.DEFAULT_RANDOM_MASK_PATH,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
    )
    parser.add_argument("--train-speeds", nargs="+", type=float, default=[2.4])
    parser.add_argument("--test-speeds", nargs="+", type=float, default=[19.0])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base.run_all_seeds(args)


if __name__ == "__main__":
    main()
