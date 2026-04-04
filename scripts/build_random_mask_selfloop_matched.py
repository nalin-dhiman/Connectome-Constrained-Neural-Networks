#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from random_mask_utils import (
    DEFAULT_RANDOM_MASK_PATH,
    init_random_mask_network,
    load_base_connectome,
    save_random_mask,
)


DEFAULT_OUTPUT_PATH = Path("results/main_results/random_mask_selfloop.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a self-loop-matched random mask control connectome."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def sample_selfloop_matched_edges(
    n_nodes: int,
    n_edges: int,
    n_self_loops: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    loop_nodes = rng.choice(n_nodes, size=n_self_loops, replace=False)
    loop_source = loop_nodes.astype(np.int64)
    loop_target = loop_nodes.astype(np.int64)

    non_loop_edges = n_edges - n_self_loops
    offdiag_capacity = n_nodes * (n_nodes - 1)
    picks = rng.choice(offdiag_capacity, size=non_loop_edges, replace=False)
    source = picks // (n_nodes - 1)
    target_offset = picks % (n_nodes - 1)
    target = target_offset + (target_offset >= source)

    random_source = np.concatenate([loop_source, source.astype(np.int64)])
    random_target = np.concatenate([loop_target, target.astype(np.int64)])
    return random_source, random_target


def build_payload(seed: int) -> dict:
    connectome = load_base_connectome()
    original_source = np.array(connectome.edges.source_index[:], copy=True)
    original_target = np.array(connectome.edges.target_index[:], copy=True)

    n_nodes = int(len(connectome.nodes.index))
    n_edges = int(len(original_source))
    original_self_loops = int(np.sum(original_source == original_target))

    random_source, random_target = sample_selfloop_matched_edges(
        n_nodes=n_nodes,
        n_edges=n_edges,
        n_self_loops=original_self_loops,
        seed=seed,
    )

    random_pairs = np.stack([random_source, random_target], axis=1)
    duplicate_edges = int(len(random_pairs) - len(np.unique(random_pairs, axis=0)))
    random_self_loops = int(np.sum(random_source == random_target))

    payload = {
        "seed": seed,
        "connectome_type": "RandomMaskConnectome",
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
        },
    }
    return payload


def main() -> None:
    args = parse_args()
    if args.output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing self-loop-matched random mask: {args.output_path}"
        )

    payload = build_payload(args.seed)
    save_random_mask(payload, args.output_path)
    network = init_random_mask_network(mask_path=args.output_path, checkpoint_path=None)

    print(f"Saved self-loop-matched random mask to {args.output_path}")
    print(f"Original self-loops: {payload['stats']['original_self_loops']}")
    print(f"Random self-loops: {payload['stats']['random_self_loops']}")
    print(f"Total edges: {payload['stats']['n_edges']}")
    print(f"Duplicate edges: {payload['stats']['duplicate_edges']}")
    print(f"Density: {payload['stats']['density']:.9f}")
    print(
        "Parameter count: "
        f"free={int(network.num_parameters.free)}, fixed={int(network.num_parameters.fixed)}"
    )

    summary_path = args.output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "mask_path": str(args.output_path),
                "free_parameters": int(network.num_parameters.free),
                "fixed_parameters": int(network.num_parameters.fixed),
                **payload["stats"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
