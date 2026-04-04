#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from degree_preserving_mask_utils import init_degree_preserving_random_network
from revision_control_utils import (
    BASELINE_CHECKPOINT,
    build_stage3_train_payload,
    ensure_dir,
    run_fixed_steps,
    value_at_iteration,
    write_json,
)


DEFAULT_OUTPUT_ROOT = Path("results/revision_results/revision_degpres_ensemble")
DEFAULT_MASK_ROOT = DEFAULT_OUTPUT_ROOT / "masks"
DEFAULT_REFERENCE_ROOT = Path("results/revision_results/revision_phase2_degree_preserving/steps_5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the degree-preserving ensemble comparison while reusing the "
            "already-completed connectome baseline from the corrected Phase 2 path."
        )
    )
    parser.add_argument("--sample-id", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-iters", type=int, default=5)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--mask-root",
        type=Path,
        default=DEFAULT_MASK_ROOT,
    )
    parser.add_argument(
        "--connectome-reference-root",
        type=Path,
        default=DEFAULT_REFERENCE_ROOT,
        help="Completed corrected Phase 2 directory used to reuse connectome curves.",
    )
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=BASELINE_CHECKPOINT,
        help="Retained for metadata parity; not loaded during from-scratch init.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip seeds whose sample-specific summary already exists.",
    )
    parser.add_argument("--train-speeds", nargs="+", type=float, default=[2.4])
    parser.add_argument("--test-speeds", nargs="+", type=float, default=[19.0])
    return parser.parse_args()


def load_connectome_reference(reference_root: Path, seed: int) -> tuple[Path, dict]:
    seed_root = reference_root / f"seed_{seed}"
    curve_path = seed_root / "connectome_curve.csv"
    summary_path = seed_root / "summary.json"
    if not curve_path.exists():
        raise FileNotFoundError(f"Missing reference connectome curve: {curve_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing reference connectome summary: {summary_path}")
    summary = json.loads(summary_path.read_text())
    return curve_path, summary


def main() -> None:
    args = parse_args()
    mask_path = args.mask_root / f"degpres_sample_{args.sample_id}.pt"
    sample_summary_path = (
        args.output_root / "summaries" / f"degpres_sample_{args.sample_id}.summary.json"
    )
    if not mask_path.exists():
        raise FileNotFoundError(f"Degree-preserving sample not found: {mask_path}")
    if not sample_summary_path.exists():
        raise FileNotFoundError(f"Degree-preserving sample summary not found: {sample_summary_path}")

    sample_root = args.output_root / f"steps_{args.max_iters}" / f"sample_{args.sample_id}"
    if sample_root.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite existing sample output: {sample_root}")

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
    batch_info["null_type"] = "degree_preserving_random_ensemble"
    batch_info["connectome_reuse_note"] = (
        "Connectome curves are copied from the existing corrected Phase 2 step-5 "
        "runs because the connectome condition is sample-invariant."
    )

    sample_root = ensure_dir(sample_root)
    sample_summary_payload = json.loads(sample_summary_path.read_text())
    all_seed_summaries = []

    print(f"Canonical train input shape: {tuple(train_stimuli.shape)}")
    print(f"Canonical train batch size: {batch_info['train_batch_size']}")
    print(f"Sample output directory: {sample_root}")
    print(f"Using degree-preserving mask: {mask_path}")

    for seed in args.seeds:
        seed_root = ensure_dir(sample_root / f"seed_{seed}")
        summary_path = seed_root / "summary.json"
        if args.resume and summary_path.exists():
            all_seed_summaries.append(json.loads(summary_path.read_text()))
            print(f"Skipping completed sample={args.sample_id} seed={seed}.")
            continue

        reference_curve_path, reference_summary = load_connectome_reference(
            args.connectome_reference_root, seed
        )
        connectome_out_curve = seed_root / "connectome_curve.csv"
        shutil.copy2(reference_curve_path, connectome_out_curve)

        degree_curve, degree_summary = run_fixed_steps(
            model_kind="degreepres",
            seed=seed,
            max_iters=args.max_iters,
            learning_rate=args.learning_rate,
            stimuli=train_stimuli,
            targets=train_targets,
            feature_start=feature_start,
            feature_stop=feature_stop,
            dt=generalization_config.dt,
            steady_state_seconds=generalization_config.steady_state_seconds,
            network_factory=lambda init_seed: init_degree_preserving_random_network(
                mask_path=mask_path,
                init_seed=init_seed,
            ),
        )
        degree_curve.to_csv(seed_root / "degreepres_curve.csv", index=False)

        seed_summary = {
            "sample_id": args.sample_id,
            "seed": seed,
            "batch_info": batch_info,
            "connectome": reference_summary["connectome"],
            "degreepres": degree_summary,
            "connectome_curve_reused_from": str(reference_curve_path),
            "connectome_summary_reused_from": str(
                args.connectome_reference_root / f"seed_{seed}" / "summary.json"
            ),
            "baseline_checkpoint_path_for_reference_only": str(args.baseline_checkpoint),
            "degree_mask_path": str(mask_path),
            "degree_mask_summary_path": str(sample_summary_path),
            "initialization_mode": "from_scratch_no_checkpoint_connectome_reused",
        }
        write_json(summary_path, seed_summary)
        all_seed_summaries.append(seed_summary)

        print(
            f"sample={args.sample_id} seed={seed} "
            f"degreepres loss@{args.max_iters}="
            f"{value_at_iteration(degree_curve, args.max_iters, 'task_loss')} "
            f"activity@{args.max_iters}="
            f"{value_at_iteration(degree_curve, args.max_iters, 'activity_abs_mean')}"
        )

    sample_level_summary = {
        "sample_id": args.sample_id,
        "max_iters": args.max_iters,
        "seeds": sorted(summary["seed"] for summary in all_seed_summaries),
        "batch_info": batch_info,
        "degree_mask_path": str(mask_path),
        "degree_mask_summary_path": str(sample_summary_path),
        "seed_summaries": all_seed_summaries,
    }
    write_json(sample_root / "summary.json", sample_level_summary)


if __name__ == "__main__":
    main()
