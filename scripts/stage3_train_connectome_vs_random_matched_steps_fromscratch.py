#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

import random_mask_utils  # Registers RandomMaskConnectome.

from revision_control_utils import (
    BASELINE_CHECKPOINT,
    FLYVIS_REPO_ROOT,
    build_stage3_train_payload,
    ensure_dir,
    run_fixed_steps,
    value_at_iteration,
    write_json,
)

if str(FLYVIS_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(FLYVIS_REPO_ROOT))

from flyvis.network.network import Network


DEFAULT_OUTPUT_ROOT = Path("results/revision_results/revision_phase1_random_init")
DEFAULT_RANDOM_MASK_PATH = Path("results/main_results/random_mask_selfloop.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Matched-step connectome vs random comparison from shared random "
            "initialization without checkpoint recovery."
        )
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-iters", type=int, default=5)
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=BASELINE_CHECKPOINT,
        help="Retained for metadata/path parity; not loaded during from-scratch init.",
    )
    parser.add_argument(
        "--random-mask-path",
        type=Path,
        default=DEFAULT_RANDOM_MASK_PATH,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume into an existing step directory by skipping seeds with summary.json.",
    )
    parser.add_argument("--train-speeds", nargs="+", type=float, default=[2.4])
    parser.add_argument("--test-speeds", nargs="+", type=float, default=[19.0])
    return parser.parse_args()


def init_connectome_factory(_baseline_checkpoint: Path):
    from revision_control_utils import init_connectome_network_from_scratch

    def factory(seed: int):
        return init_connectome_network_from_scratch(seed)

    return factory


def init_random_factory(mask_path: Path):
    def factory(seed: int):
        from revision_control_utils import base_network_config

        config = base_network_config(seed)
        config.network.connectome.type = "RandomMaskConnectome"
        config.network.connectome.mask_path = str(mask_path)
        for key in ["file", "extent", "n_syn_fill"]:
            if key in config.network.connectome:
                del config.network.connectome[key]
        return Network(**config.network)

    return factory


def main() -> None:
    args = parse_args()
    step_root = args.output_root / f"steps_{args.max_iters}"
    if step_root.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite existing revision output: {step_root}")
    if not args.random_mask_path.exists():
        raise FileNotFoundError(f"Random mask not found: {args.random_mask_path}")
    payload = build_stage3_train_payload(
        train_speeds=args.train_speeds,
        test_speeds=args.test_speeds,
    )
    train_stimuli = payload["stimuli"]
    train_targets = payload["targets"]
    feature_start = payload["feature_start"]
    feature_stop = payload["feature_stop"]
    generalization_config = payload["generalization_config"]
    batch_info = payload["batch_info"]
    batch_info["max_iters"] = int(args.max_iters)
    batch_info["init_confounds_removed"] = [
        "No recover_network() checkpoint loading",
        "Node-bias initialization seed tied to experiment seed",
        "Decoder initialization tied to experiment seed",
    ]
    print(f"Canonical train input shape: {tuple(train_stimuli.shape)}")
    print(f"Canonical train batch size: {batch_info['train_batch_size']}")
    print(f"Output directory: {step_root}")

    output_root = ensure_dir(step_root)
    seed_summaries: list[dict] = []

    connectome_factory = init_connectome_factory(args.baseline_checkpoint)
    random_factory = init_random_factory(args.random_mask_path)

    for seed in args.seeds:
        seed_root = ensure_dir(output_root / f"seed_{seed}")
        summary_path = seed_root / "summary.json"
        if args.resume and summary_path.exists():
            seed_summaries.append(json.loads(summary_path.read_text()))
            print(f"Skipping completed seed={seed} because {summary_path} already exists.")
            continue
        connectome_curve, connectome_summary = run_fixed_steps(
            model_kind="connectome",
            seed=seed,
            max_iters=args.max_iters,
            learning_rate=args.learning_rate,
            stimuli=train_stimuli,
            targets=train_targets,
            feature_start=feature_start,
            feature_stop=feature_stop,
            dt=generalization_config.dt,
            steady_state_seconds=generalization_config.steady_state_seconds,
            network_factory=connectome_factory,
        )
        random_curve, random_summary = run_fixed_steps(
            model_kind="random",
            seed=seed,
            max_iters=args.max_iters,
            learning_rate=args.learning_rate,
            stimuli=train_stimuli,
            targets=train_targets,
            feature_start=feature_start,
            feature_stop=feature_stop,
            dt=generalization_config.dt,
            steady_state_seconds=generalization_config.steady_state_seconds,
            network_factory=random_factory,
        )

        connectome_curve.to_csv(seed_root / "connectome_curve.csv", index=False)
        random_curve.to_csv(seed_root / "random_curve.csv", index=False)

        seed_summary = {
            "seed": seed,
            "batch_info": batch_info,
            "connectome": connectome_summary,
            "random": random_summary,
            "baseline_checkpoint_path_for_reference_only": str(args.baseline_checkpoint),
            "random_mask_path": str(args.random_mask_path),
            "initialization_mode": "from_scratch_no_checkpoint",
        }
        write_json(seed_root / "summary.json", seed_summary)
        seed_summaries.append(seed_summary)

        print(
            f"seed={seed} connectome loss@{args.max_iters}="
            f"{value_at_iteration(connectome_curve, args.max_iters, 'task_loss')} "
            f"activity@{args.max_iters}="
            f"{value_at_iteration(connectome_curve, args.max_iters, 'activity_abs_mean')}"
        )
        print(
            f"seed={seed} random loss@{args.max_iters}="
            f"{value_at_iteration(random_curve, args.max_iters, 'task_loss')} "
            f"activity@{args.max_iters}="
            f"{value_at_iteration(random_curve, args.max_iters, 'activity_abs_mean')}"
        )

    all_seed_summaries = []
    for summary_path in sorted(output_root.glob("seed_*/summary.json")):
        all_seed_summaries.append(json.loads(summary_path.read_text()))

    write_json(
        output_root / "summary.json",
        {
            "phase": "phase1_random_init",
            "max_iters": args.max_iters,
            "seeds": sorted(summary["seed"] for summary in all_seed_summaries),
            "batch_info": batch_info,
            "seed_summaries": all_seed_summaries,
        },
    )


if __name__ == "__main__":
    main()
