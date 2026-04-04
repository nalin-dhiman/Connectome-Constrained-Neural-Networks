#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from random_mask_utils import load_base_connectome, save_random_mask
from ringlattice_mask_utils import (
    DEFAULT_RINGLATTICE_MASK_PATH,
    init_ringlattice_mask_network,
)


DEFAULT_OUTPUT_PATH = DEFAULT_RINGLATTICE_MASK_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a self-loop-matched ring-lattice control connectome."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def sample_ringlattice_edges(
    *,
    n_nodes: int,
    n_edges: int,
    n_self_loops: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    loop_nodes = rng.choice(n_nodes, size=n_self_loops, replace=False)
    loop_source = loop_nodes.astype(np.int64)
    loop_target = loop_nodes.astype(np.int64)

    nonloop_edges = n_edges - n_self_loops
    base_out_degree = nonloop_edges // n_nodes
    remainder = nonloop_edges % n_nodes
    out_degree = np.full(n_nodes, base_out_degree, dtype=np.int64)
    out_degree[:remainder] += 1

    source = np.repeat(np.arange(n_nodes, dtype=np.int64), out_degree)
    target = np.empty(nonloop_edges, dtype=np.int64)
    position = 0
    for node, degree in enumerate(out_degree):
        if degree == 0:
            continue
        offsets = np.arange(1, degree + 1, dtype=np.int64)
        target[position : position + degree] = (node + offsets) % n_nodes
        position += degree

    ring_source = np.concatenate([loop_source, source])
    ring_target = np.concatenate([loop_target, target.astype(np.int64)])
    return ring_source, ring_target


def build_payload(seed: int) -> dict:
    connectome = load_base_connectome()
    original_source = np.array(connectome.edges.source_index[:], copy=True)
    original_target = np.array(connectome.edges.target_index[:], copy=True)

    n_nodes = int(len(connectome.nodes.index))
    n_edges = int(len(original_source))
    original_self_loops = int(np.sum(original_source == original_target))

    ring_source, ring_target = sample_ringlattice_edges(
        n_nodes=n_nodes,
        n_edges=n_edges,
        n_self_loops=original_self_loops,
        seed=seed,
    )

    pairs = np.stack([ring_source, ring_target], axis=1)
    duplicate_edges = int(len(pairs) - len(np.unique(pairs, axis=0)))
    ringlattice_self_loops = int(np.sum(ring_source == ring_target))
    in_degree = np.bincount(ring_target, minlength=n_nodes)
    out_degree = np.bincount(ring_source, minlength=n_nodes)

    node_u = np.array(connectome.nodes.u[:], copy=True)
    node_v = np.array(connectome.nodes.v[:], copy=True)

    payload = {
        "seed": seed,
        "connectome_type": "RingLatticeMaskConnectome",
        "nodes": {
            "index": np.array(connectome.nodes.index[:], copy=True),
            "type": np.array(connectome.nodes.type[:], copy=True),
            "u": np.array(connectome.nodes.u[:], copy=True),
            "v": np.array(connectome.nodes.v[:], copy=True),
            "role": np.array(connectome.nodes.role[:], copy=True),
        },
        "edges": {
            "source_index": ring_source.astype(np.int64),
            "target_index": ring_target.astype(np.int64),
            "sign": np.array(connectome.edges.sign[:], copy=True),
            "n_syn": np.array(connectome.edges.n_syn[:], copy=True),
            "source_type": np.array(connectome.edges.source_type[:], copy=True),
            "target_type": np.array(connectome.edges.target_type[:], copy=True),
            "source_u": node_u[ring_source].astype(np.int32),
            "target_u": node_u[ring_target].astype(np.int32),
            "source_v": node_v[ring_source].astype(np.int32),
            "target_v": node_v[ring_target].astype(np.int32),
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
            "ringlattice_self_loops": ringlattice_self_loops,
            "duplicate_edges": duplicate_edges,
            "in_degree_mean": float(in_degree.mean()),
            "in_degree_std": float(in_degree.std()),
            "in_degree_min": int(in_degree.min()),
            "in_degree_max": int(in_degree.max()),
            "out_degree_mean": float(out_degree.mean()),
            "out_degree_std": float(out_degree.std()),
            "out_degree_min": int(out_degree.min()),
            "out_degree_max": int(out_degree.max()),
            "remainder_rule": (
                "Assigned one extra outgoing lattice edge to the earliest nodes "
                "when (E-L) was not divisible by N."
            ),
        },
    }
    return payload


def main() -> None:
    args = parse_args()
    if args.output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing ring-lattice mask: {args.output_path}"
        )

    payload = build_payload(args.seed)
    if payload["stats"]["n_edges"] != len(payload["edges"]["source_index"]):
        raise RuntimeError("Ring-lattice edge count mismatch.")
    if payload["stats"]["ringlattice_self_loops"] != payload["stats"]["original_self_loops"]:
        raise RuntimeError("Ring-lattice self-loop count mismatch.")
    if payload["stats"]["duplicate_edges"] != 0:
        raise RuntimeError("Ring-lattice mask has duplicate edges.")

    save_random_mask(payload, args.output_path)
    network = init_ringlattice_mask_network(mask_path=args.output_path)
    if int(network.n_nodes) != int(payload["stats"]["n_nodes"]):
        raise RuntimeError("Ring-lattice node count mismatch after init.")
    if int(network.n_edges) != int(payload["stats"]["n_edges"]):
        raise RuntimeError("Ring-lattice edge count mismatch after init.")

    print(f"Saved ring-lattice mask to {args.output_path}")
    print(f"Nodes: {payload['stats']['n_nodes']}")
    print(f"Edges: {payload['stats']['n_edges']}")
    print(f"Original self-loops: {payload['stats']['original_self_loops']}")
    print(f"Ring-lattice self-loops: {payload['stats']['ringlattice_self_loops']}")
    print(f"Duplicate edges: {payload['stats']['duplicate_edges']}")
    print(
        "Degree stats: "
        f"in(mean={payload['stats']['in_degree_mean']:.3f}, std={payload['stats']['in_degree_std']:.3f}), "
        f"out(mean={payload['stats']['out_degree_mean']:.3f}, std={payload['stats']['out_degree_std']:.3f})"
    )
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
