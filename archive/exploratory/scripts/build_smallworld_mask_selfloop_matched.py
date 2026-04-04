#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from random_mask_utils import load_base_connectome, save_random_mask
from smallworld_mask_utils import (
    DEFAULT_SMALLWORLD_MASK_PATH,
    init_smallworld_mask_network,
)


DEFAULT_REWIRE_PROB = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a self-loop-matched small-world mask control connectome."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rewire-prob", type=float, default=DEFAULT_REWIRE_PROB)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_SMALLWORLD_MASK_PATH)
    return parser.parse_args()


def directed_ring_lattice_targets(
    n_nodes: int,
    n_nonloop_edges: int,
) -> tuple[np.ndarray, np.ndarray]:
    base_out_degree = n_nonloop_edges // n_nodes
    remainder = n_nonloop_edges % n_nodes
    out_degree = np.full(n_nodes, base_out_degree, dtype=np.int64)
    out_degree[:remainder] += 1

    source = np.repeat(np.arange(n_nodes, dtype=np.int64), out_degree)
    target = np.empty(n_nonloop_edges, dtype=np.int64)
    position = 0
    for node, degree in enumerate(out_degree):
        if degree == 0:
            continue
        offsets = np.arange(1, degree + 1, dtype=np.int64)
        target[position : position + degree] = (node + offsets) % n_nodes
        position += degree
    return source, target


def rewire_smallworld_edges(
    source: np.ndarray,
    target: np.ndarray,
    *,
    n_nodes: int,
    rewire_prob: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    rewired_target = target.copy()
    out_neighbors = [set() for _ in range(n_nodes)]
    positions_by_source: list[list[int]] = [[] for _ in range(n_nodes)]

    for idx, (src, dst) in enumerate(zip(source.tolist(), rewired_target.tolist())):
        out_neighbors[src].add(dst)
        positions_by_source[src].append(idx)

    for src in range(n_nodes):
        for idx in positions_by_source[src]:
            if rng.random() >= rewire_prob:
                continue
            old_dst = int(rewired_target[idx])
            out_neighbors[src].remove(old_dst)
            while True:
                candidate = int(rng.integers(0, n_nodes))
                if candidate == src:
                    continue
                if candidate in out_neighbors[src]:
                    continue
                rewired_target[idx] = candidate
                out_neighbors[src].add(candidate)
                break

    return source, rewired_target


def sample_smallworld_edges(
    *,
    n_nodes: int,
    n_edges: int,
    n_self_loops: int,
    seed: int,
    rewire_prob: float,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    loop_nodes = rng.choice(n_nodes, size=n_self_loops, replace=False)
    loop_source = loop_nodes.astype(np.int64)
    loop_target = loop_nodes.astype(np.int64)

    nonloop_edges = n_edges - n_self_loops
    nonloop_source, nonloop_target = directed_ring_lattice_targets(n_nodes, nonloop_edges)
    nonloop_source, nonloop_target = rewire_smallworld_edges(
        nonloop_source,
        nonloop_target,
        n_nodes=n_nodes,
        rewire_prob=rewire_prob,
        rng=rng,
    )

    source = np.concatenate([loop_source, nonloop_source])
    target = np.concatenate([loop_target, nonloop_target])
    return source, target


def build_payload(seed: int, rewire_prob: float) -> dict:
    connectome = load_base_connectome()
    original_source = np.array(connectome.edges.source_index[:], copy=True)
    original_target = np.array(connectome.edges.target_index[:], copy=True)

    n_nodes = int(len(connectome.nodes.index))
    n_edges = int(len(original_source))
    original_self_loops = int(np.sum(original_source == original_target))

    source, target = sample_smallworld_edges(
        n_nodes=n_nodes,
        n_edges=n_edges,
        n_self_loops=original_self_loops,
        seed=seed,
        rewire_prob=rewire_prob,
    )

    node_u = np.array(connectome.nodes.u[:], copy=True)
    node_v = np.array(connectome.nodes.v[:], copy=True)
    pairs = np.stack([source, target], axis=1)
    duplicate_edges = int(len(pairs) - len(np.unique(pairs, axis=0)))
    smallworld_self_loops = int(np.sum(source == target))
    in_degree = np.bincount(target, minlength=n_nodes)
    out_degree = np.bincount(source, minlength=n_nodes)

    payload = {
        "seed": seed,
        "connectome_type": "SmallWorldMaskConnectome",
        "nodes": {
            "index": np.array(connectome.nodes.index[:], copy=True),
            "type": np.array(connectome.nodes.type[:], copy=True),
            "u": np.array(connectome.nodes.u[:], copy=True),
            "v": np.array(connectome.nodes.v[:], copy=True),
            "role": np.array(connectome.nodes.role[:], copy=True),
        },
        "edges": {
            "source_index": source.astype(np.int64),
            "target_index": target.astype(np.int64),
            "sign": np.array(connectome.edges.sign[:], copy=True),
            "n_syn": np.array(connectome.edges.n_syn[:], copy=True),
            "source_type": np.array(connectome.edges.source_type[:], copy=True),
            "target_type": np.array(connectome.edges.target_type[:], copy=True),
            "source_u": node_u[source].astype(np.int32),
            "target_u": node_u[target].astype(np.int32),
            "source_v": node_v[source].astype(np.int32),
            "target_v": node_v[target].astype(np.int32),
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
            "smallworld_self_loops": smallworld_self_loops,
            "duplicate_edges": duplicate_edges,
            "in_degree_mean": float(in_degree.mean()),
            "in_degree_std": float(in_degree.std()),
            "in_degree_min": int(in_degree.min()),
            "in_degree_max": int(in_degree.max()),
            "out_degree_mean": float(out_degree.mean()),
            "out_degree_std": float(out_degree.std()),
            "out_degree_min": int(out_degree.min()),
            "out_degree_max": int(out_degree.max()),
            "rewire_prob": float(rewire_prob),
        },
    }
    return payload


def main() -> None:
    args = parse_args()
    if args.output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing self-loop-matched small-world mask: {args.output_path}"
        )

    payload = build_payload(args.seed, args.rewire_prob)
    if payload["stats"]["duplicate_edges"] != 0:
        raise RuntimeError("Small-world mask has duplicate edges.")
    if payload["stats"]["smallworld_self_loops"] != payload["stats"]["original_self_loops"]:
        raise RuntimeError("Small-world mask self-loop count does not match connectome.")

    save_random_mask(payload, args.output_path)
    network = init_smallworld_mask_network(mask_path=args.output_path)

    print(f"Saved small-world mask to {args.output_path}")
    print(f"Nodes: {payload['stats']['n_nodes']}")
    print(f"Edges: {payload['stats']['n_edges']}")
    print(f"Original self-loops: {payload['stats']['original_self_loops']}")
    print(f"Small-world self-loops: {payload['stats']['smallworld_self_loops']}")
    print(f"Duplicate edges: {payload['stats']['duplicate_edges']}")
    print(
        "Degree stats: "
        f"in(mean={payload['stats']['in_degree_mean']:.3f}, std={payload['stats']['in_degree_std']:.3f}, "
        f"min={payload['stats']['in_degree_min']}, max={payload['stats']['in_degree_max']}), "
        f"out(mean={payload['stats']['out_degree_mean']:.3f}, std={payload['stats']['out_degree_std']:.3f}, "
        f"min={payload['stats']['out_degree_min']}, max={payload['stats']['out_degree_max']})"
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
