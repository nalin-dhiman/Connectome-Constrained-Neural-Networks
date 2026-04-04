#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from random_mask_utils import load_base_connectome, save_random_mask


DEFAULT_OUTPUT_PATH = Path("results/revision_results/degree_preserving_random_mask.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a degree-preserving random mask via directed double-edge swaps "
            "while keeping self-loops fixed."
        )
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--n-swaps",
        type=int,
        default=250000,
        help="Number of accepted non-loop edge swaps to perform.",
    )
    parser.add_argument(
        "--max-attempt-factor",
        type=int,
        default=20,
        help="Maximum attempts as a multiple of requested accepted swaps.",
    )
    return parser.parse_args()


def edge_codes(source: np.ndarray, target: np.ndarray, n_nodes: int) -> np.ndarray:
    return source.astype(np.int64) * np.int64(n_nodes) + target.astype(np.int64)


def degree_stats(source: np.ndarray, target: np.ndarray, n_nodes: int) -> dict[str, float]:
    in_deg = np.bincount(target, minlength=n_nodes)
    out_deg = np.bincount(source, minlength=n_nodes)
    return {
        "in_degree_mean": float(np.mean(in_deg)),
        "in_degree_std": float(np.std(in_deg)),
        "in_degree_min": int(np.min(in_deg)),
        "in_degree_max": int(np.max(in_deg)),
        "out_degree_mean": float(np.mean(out_deg)),
        "out_degree_std": float(np.std(out_deg)),
        "out_degree_min": int(np.min(out_deg)),
        "out_degree_max": int(np.max(out_deg)),
    }


def swap_degree_preserving_edges(
    source: np.ndarray,
    target: np.ndarray,
    *,
    n_nodes: int,
    rng: np.random.Generator,
    n_swaps: int,
    max_attempt_factor: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    source = source.astype(np.int64, copy=True)
    target = target.astype(np.int64, copy=True)
    non_loop_mask = source != target
    non_loop_indices = np.flatnonzero(non_loop_mask)
    codes = set(edge_codes(source, target, n_nodes).tolist())

    accepted = 0
    attempts = 0
    max_attempts = int(max(1, n_swaps * max_attempt_factor))

    while accepted < n_swaps and attempts < max_attempts:
        attempts += 1
        i1, i2 = rng.choice(non_loop_indices, size=2, replace=False)
        a = int(source[i1])
        b = int(target[i1])
        c = int(source[i2])
        d = int(target[i2])

        if len({a, b, c, d}) < 4:
            continue

        new1 = (a, d)
        new2 = (c, b)
        if new1[0] == new1[1] or new2[0] == new2[1]:
            continue
        if new1 == new2:
            continue

        old_code1 = a * n_nodes + b
        old_code2 = c * n_nodes + d
        new_code1 = new1[0] * n_nodes + new1[1]
        new_code2 = new2[0] * n_nodes + new2[1]

        if new_code1 in codes or new_code2 in codes:
            continue

        codes.remove(old_code1)
        codes.remove(old_code2)
        codes.add(new_code1)
        codes.add(new_code2)

        target[i1] = new1[1]
        target[i2] = new2[1]
        accepted += 1

    return source, target, {
        "accepted_swaps": int(accepted),
        "attempted_swaps": int(attempts),
        "max_attempts": int(max_attempts),
    }


def build_payload(seed: int, n_swaps: int, max_attempt_factor: int) -> dict:
    connectome = load_base_connectome()
    original_source = np.array(connectome.edges.source_index[:], copy=True)
    original_target = np.array(connectome.edges.target_index[:], copy=True)
    n_nodes = int(len(connectome.nodes.index))
    n_edges = int(len(original_source))
    original_self_loops = int(np.sum(original_source == original_target))

    rng = np.random.default_rng(seed)
    random_source, random_target, swap_stats = swap_degree_preserving_edges(
        original_source,
        original_target,
        n_nodes=n_nodes,
        rng=rng,
        n_swaps=n_swaps,
        max_attempt_factor=max_attempt_factor,
    )

    pair_codes = edge_codes(random_source, random_target, n_nodes)
    duplicate_edges = int(len(pair_codes) - len(np.unique(pair_codes)))
    random_self_loops = int(np.sum(random_source == random_target))

    if duplicate_edges != 0:
        raise RuntimeError(
            f"Degree-preserving construction produced duplicate edges: {duplicate_edges}"
        )
    if random_self_loops != original_self_loops:
        raise RuntimeError(
            "Degree-preserving construction changed self-loop count: "
            f"{random_self_loops} vs {original_self_loops}"
        )
    if len(random_source) != n_edges:
        raise RuntimeError("Edge count changed during degree-preserving construction.")

    orig_in = np.bincount(original_target, minlength=n_nodes)
    orig_out = np.bincount(original_source, minlength=n_nodes)
    rand_in = np.bincount(random_target, minlength=n_nodes)
    rand_out = np.bincount(random_source, minlength=n_nodes)
    if not np.array_equal(orig_in, rand_in):
        raise RuntimeError("In-degree sequence is not preserved exactly.")
    if not np.array_equal(orig_out, rand_out):
        raise RuntimeError("Out-degree sequence is not preserved exactly.")

    payload = {
        "seed": seed,
        "connectome_type": "DegreePreservingRandomConnectome",
        "nodes": {
            "index": np.array(connectome.nodes.index[:], copy=True),
            "type": np.array(connectome.nodes.type[:], copy=True),
            "u": np.array(connectome.nodes.u[:], copy=True),
            "v": np.array(connectome.nodes.v[:], copy=True),
            "role": np.array(connectome.nodes.role[:], copy=True),
        },
        "edges": {
            "source_index": random_source.astype(np.int64),
            "target_index": random_target.astype(np.int64),
            "sign": np.array(connectome.edges.sign[:], copy=True),
            "n_syn": np.array(connectome.edges.n_syn[:], copy=True),
            "source_type": np.array(connectome.edges.source_type[:], copy=True),
            "target_type": np.array(connectome.edges.target_type[:], copy=True),
            "source_u": np.array(connectome.edges.source_u[:], copy=True),
            "target_u": np.array(connectome.edges.target_u[:], copy=True),
            "source_v": np.array(connectome.edges.source_v[:], copy=True),
            "target_v": np.array(connectome.edges.target_v[:], copy=True),
            "du": np.array(connectome.edges.du[:], copy=True),
            "dv": np.array(connectome.edges.dv[:], copy=True),
            "n_syn_certainty": np.array(connectome.edges.n_syn_certainty[:], copy=True),
        },
        "layer_index": {
            key: np.array(value[:], copy=True)
            for key, value in connectome.nodes.layer_index.items()
        },
        "unique_cell_types": np.array(connectome.unique_cell_types[:], copy=True),
        "input_cell_types": np.array(connectome.input_cell_types[:], copy=True),
        "output_cell_types": np.array(connectome.output_cell_types[:], copy=True),
        "intermediate_cell_types": np.array(
            connectome.intermediate_cell_types[:], copy=True
        ),
        "layout": np.array(connectome.layout[:], copy=True),
        "central_cells_index": np.array(connectome.central_cells_index[:], copy=True),
        "stats": {
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "density": float(n_edges / (n_nodes * n_nodes)),
            "original_self_loops": original_self_loops,
            "random_self_loops": random_self_loops,
            "duplicate_edges": duplicate_edges,
            "unchanged_edge_fraction": float(
                np.mean(
                    (random_source == original_source) & (random_target == original_target)
                )
            ),
            **swap_stats,
            **degree_stats(random_source, random_target, n_nodes),
        },
    }
    payload["stats"]["in_degree_exact_match"] = True
    payload["stats"]["out_degree_exact_match"] = True
    return payload


def main() -> None:
    args = parse_args()
    if args.output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing degree-preserving mask: {args.output_path}"
        )

    payload = build_payload(args.seed, args.n_swaps, args.max_attempt_factor)
    save_random_mask(payload, args.output_path)

    summary_path = args.output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "mask_path": str(args.output_path),
                **payload["stats"],
            },
            indent=2,
            sort_keys=True,
        )
    )

    print(f"Saved degree-preserving random mask to {args.output_path}")
    print(f"Nodes: {payload['stats']['n_nodes']}")
    print(f"Edges: {payload['stats']['n_edges']}")
    print(f"Self-loops: {payload['stats']['random_self_loops']}")
    print(f"Duplicate edges: {payload['stats']['duplicate_edges']}")
    print(
        "In-degree stats "
        f"mean/std/min/max={payload['stats']['in_degree_mean']:.4f}/"
        f"{payload['stats']['in_degree_std']:.4f}/"
        f"{payload['stats']['in_degree_min']}/"
        f"{payload['stats']['in_degree_max']}"
    )
    print(
        "Out-degree stats "
        f"mean/std/min/max={payload['stats']['out_degree_mean']:.4f}/"
        f"{payload['stats']['out_degree_std']:.4f}/"
        f"{payload['stats']['out_degree_min']}/"
        f"{payload['stats']['out_degree_max']}"
    )
    print(
        f"Accepted swaps: {payload['stats']['accepted_swaps']} / "
        f"attempts: {payload['stats']['attempted_swaps']}"
    )
    print(f"Unchanged edge fraction: {payload['stats']['unchanged_edge_fraction']:.9f}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
