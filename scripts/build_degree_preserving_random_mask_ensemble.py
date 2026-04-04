#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_degree_preserving_random_mask import build_payload
from random_mask_utils import save_random_mask


DEFAULT_OUTPUT_ROOT = Path("results/revision_results/revision_degpres_ensemble")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an ensemble of degree-preserving random masks by reusing the "
            "validated directed double-edge swap pipeline."
        )
    )
    parser.add_argument("--k", type=int, default=5, help="Number of masks to generate.")
    parser.add_argument(
        "--base-seed",
        type=int,
        default=0,
        help="Sample i uses seed base_seed + i.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--n-swaps",
        type=int,
        default=250000,
        help="Number of accepted non-loop edge swaps per sample.",
    )
    parser.add_argument(
        "--max-attempt-factor",
        type=int,
        default=20,
        help="Maximum attempts as a multiple of requested accepted swaps.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing ensemble output root: {args.output_root}"
        )

    masks_root = args.output_root / "masks"
    summaries_root = args.output_root / "summaries"
    masks_root.mkdir(parents=True, exist_ok=False)
    summaries_root.mkdir(parents=True, exist_ok=False)

    manifest = []
    for sample_id in range(args.k):
        seed = args.base_seed + sample_id
        payload = build_payload(seed, args.n_swaps, args.max_attempt_factor)
        stats = payload["stats"]
        if not stats.get("in_degree_exact_match", False):
            raise RuntimeError(f"Sample {sample_id} failed exact in-degree preservation.")
        if not stats.get("out_degree_exact_match", False):
            raise RuntimeError(f"Sample {sample_id} failed exact out-degree preservation.")
        if int(stats["duplicate_edges"]) != 0:
            raise RuntimeError(f"Sample {sample_id} contains duplicate edges.")
        if int(stats["original_self_loops"]) != int(stats["random_self_loops"]):
            raise RuntimeError(f"Sample {sample_id} changed self-loop count.")

        mask_path = masks_root / f"degpres_sample_{sample_id}.pt"
        summary_path = summaries_root / f"degpres_sample_{sample_id}.summary.json"
        save_random_mask(payload, mask_path)

        summary = {
            "sample_id": sample_id,
            "seed": seed,
            "mask_path": str(mask_path),
            "rewiring_method": "directed_double_edge_swap_with_fixed_self_loops",
            "preserves": {
                "n_nodes": True,
                "n_edges": True,
                "self_loops": True,
                "in_degree_exact": True,
                "out_degree_exact": True,
                "duplicates_avoided": True,
            },
            "sampler_metadata": {
                "n_swaps_requested": int(args.n_swaps),
                "max_attempt_factor": int(args.max_attempt_factor),
            },
            **stats,
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        manifest.append(summary)

        print(
            f"sample={sample_id} seed={seed} "
            f"nodes={stats['n_nodes']} edges={stats['n_edges']} "
            f"self_loops={stats['random_self_loops']} duplicates={stats['duplicate_edges']} "
            f"accepted_swaps={stats['accepted_swaps']}"
        )

    manifest_path = args.output_root / "ensemble_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Saved {args.k} degree-preserving masks under {masks_root}")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
