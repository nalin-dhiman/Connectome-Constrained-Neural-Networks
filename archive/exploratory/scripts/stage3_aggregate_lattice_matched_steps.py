#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from random_mask_utils import load_base_connectome, load_random_mask_payload


MODEL_KINDS = ("connectome", "random", "ringlattice")
STEP = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate connectome vs ring-lattice matched-step results."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("results/stage3_connectome_vs_lattice_matched_steps/steps_5"),
    )
    parser.add_argument(
        "--aggregated-csv",
        type=Path,
        default=Path("results/stage3_connectome_vs_lattice_matched_steps/steps_5/aggregated_metrics.csv"),
    )
    parser.add_argument(
        "--delta-csv",
        type=Path,
        default=Path("results/stage3_connectome_vs_lattice_matched_steps/steps_5/delta_metrics.csv"),
    )
    parser.add_argument(
        "--bootstrap-csv",
        type=Path,
        default=Path("results/stage3_connectome_vs_lattice_matched_steps/steps_5/bootstrap_iter5.csv"),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/stage3_connectome_vs_lattice_matched_steps.md"),
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    return parser.parse_args()


def fmt(mean: float, std: float, n: int) -> str:
    if n == 0 or np.isnan(mean):
        return "n/a"
    if n == 1 or np.isnan(std):
        return f"{mean:.4f} (n=1)"
    return f"{mean:.4f} +/- {std:.4f}"


def seed_dirs(root: Path) -> list[Path]:
    dirs = sorted(path for path in root.glob("seed_*") if path.is_dir())
    if not dirs:
        raise FileNotFoundError(f"No seed directories found under {root}")
    return dirs


def load_curve(seed_dir: Path, model_kind: str) -> pd.DataFrame:
    path = seed_dir / f"{model_kind}_curve.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing curve file: {path}")
    return pd.read_csv(path)


def value_at_iteration(curve_df: pd.DataFrame, iteration: int, column: str) -> float:
    row = curve_df.loc[curve_df["iteration"].eq(iteration)]
    if row.empty:
        return float("nan")
    return float(row.iloc[0][column])


def degree_stats_from_edges(source: np.ndarray, target: np.ndarray, n_nodes: int) -> dict[str, float]:
    in_degree = np.bincount(target, minlength=n_nodes)
    out_degree = np.bincount(source, minlength=n_nodes)
    return {
        "in_mean": float(in_degree.mean()),
        "in_std": float(in_degree.std()),
        "out_mean": float(out_degree.mean()),
        "out_std": float(out_degree.std()),
    }


def bootstrap_ci(values: np.ndarray, resamples: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(0)
    means = np.empty(resamples, dtype=float)
    for idx in range(resamples):
        sample = rng.choice(values, size=len(values), replace=True)
        means[idx] = float(np.mean(sample))
    return (
        float(np.mean(values)),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def main() -> None:
    args = parse_args()
    for output_path in (
        args.aggregated_csv,
        args.delta_csv,
        args.bootstrap_csv,
        args.report_path,
    ):
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")

    raw_rows: list[dict] = []
    delta_rows: list[dict] = []

    for seed_dir in seed_dirs(args.input_root):
        seed = int(seed_dir.name.split("_")[-1])
        curves = {model_kind: load_curve(seed_dir, model_kind) for model_kind in MODEL_KINDS}

        for model_kind in MODEL_KINDS:
            curve = curves[model_kind]
            raw_rows.append(
                {
                    "seed": seed,
                    "model_kind": model_kind,
                    "loss": value_at_iteration(curve, STEP, "task_loss"),
                    "activity": value_at_iteration(curve, STEP, "activity_abs_mean"),
                    "elapsed": value_at_iteration(curve, STEP, "elapsed_sec"),
                    "finite": bool(curve["finite"].all()),
                }
            )

        connectome_curve = curves["connectome"]
        random_curve = curves["random"]
        ring_curve = curves["ringlattice"]
        delta_rows.append(
            {
                "seed": seed,
                "delta_loss_random": value_at_iteration(random_curve, STEP, "task_loss")
                - value_at_iteration(connectome_curve, STEP, "task_loss"),
                "delta_activity_random": value_at_iteration(random_curve, STEP, "activity_abs_mean")
                - value_at_iteration(connectome_curve, STEP, "activity_abs_mean"),
                "delta_time_random": value_at_iteration(random_curve, STEP, "elapsed_sec")
                - value_at_iteration(connectome_curve, STEP, "elapsed_sec"),
                "delta_loss_lattice": value_at_iteration(ring_curve, STEP, "task_loss")
                - value_at_iteration(connectome_curve, STEP, "task_loss"),
                "delta_activity_lattice": value_at_iteration(ring_curve, STEP, "activity_abs_mean")
                - value_at_iteration(connectome_curve, STEP, "activity_abs_mean"),
                "delta_time_lattice": value_at_iteration(ring_curve, STEP, "elapsed_sec")
                - value_at_iteration(connectome_curve, STEP, "elapsed_sec"),
            }
        )

    raw_df = pd.DataFrame(raw_rows).sort_values(["model_kind", "seed"]).reset_index(drop=True)
    delta_df = pd.DataFrame(delta_rows).sort_values("seed").reset_index(drop=True)

    summary_rows = []
    for model_kind in MODEL_KINDS:
        model_df = raw_df.loc[raw_df["model_kind"].eq(model_kind)].reset_index(drop=True)
        summary_rows.append(
            {
                "model_kind": model_kind,
                "loss_mean": float(model_df["loss"].mean()),
                "loss_std": float(model_df["loss"].std(ddof=1)),
                "activity_mean": float(model_df["activity"].mean()),
                "activity_std": float(model_df["activity"].std(ddof=1)),
                "elapsed_mean": float(model_df["elapsed"].mean()),
                "elapsed_std": float(model_df["elapsed"].std(ddof=1)),
                "finite_all": bool(model_df["finite"].all()),
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    bootstrap_rows = []
    for metric in (
        "delta_loss_random",
        "delta_activity_random",
        "delta_time_random",
        "delta_loss_lattice",
        "delta_activity_lattice",
        "delta_time_lattice",
    ):
        values = delta_df[metric].to_numpy(dtype=float)
        mean, ci_low, ci_high = bootstrap_ci(values, args.bootstrap_resamples)
        bootstrap_rows.append(
            {"metric": metric, "mean": mean, "ci_low": ci_low, "ci_high": ci_high}
        )
    bootstrap_df = pd.DataFrame(bootstrap_rows)

    args.aggregated_csv.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(args.input_root / "raw_metrics.csv", index=False)
    summary_df.to_csv(args.aggregated_csv, index=False)
    delta_df.to_csv(args.delta_csv, index=False)
    bootstrap_df.to_csv(args.bootstrap_csv, index=False)

    connectome = load_base_connectome()
    connectome_stats = degree_stats_from_edges(
        np.array(connectome.edges.source_index[:], copy=True),
        np.array(connectome.edges.target_index[:], copy=True),
        int(len(connectome.nodes.index)),
    )
    random_payload = load_random_mask_payload("results/random_mask_selfloop.pt")
    ring_payload = load_random_mask_payload("results/ringlattice_mask_selfloop.pt")
    random_stats = degree_stats_from_edges(
        random_payload["edges"]["source_index"],
        random_payload["edges"]["target_index"],
        int(random_payload["stats"]["n_nodes"]),
    )
    ring_stats = degree_stats_from_edges(
        ring_payload["edges"]["source_index"],
        ring_payload["edges"]["target_index"],
        int(ring_payload["stats"]["n_nodes"]),
    )

    model_table = {
        "connectome": {
            "nodes": int(len(connectome.nodes.index)),
            "edges": int(len(connectome.edges.source_index)),
            "self_loops": int(np.sum(connectome.edges.source_index[:] == connectome.edges.target_index[:])),
            "free_parameters": 734,
            "fixed_parameters": 2959,
        },
        "random": {
            "nodes": int(random_payload["stats"]["n_nodes"]),
            "edges": int(random_payload["stats"]["n_edges"]),
            "self_loops": int(random_payload["stats"]["random_self_loops"]),
            "free_parameters": 734,
            "fixed_parameters": 2959,
        },
        "ringlattice": {
            "nodes": int(ring_payload["stats"]["n_nodes"]),
            "edges": int(ring_payload["stats"]["n_edges"]),
            "self_loops": int(ring_payload["stats"]["ringlattice_self_loops"]),
            "free_parameters": 734,
            "fixed_parameters": 2959,
        },
    }

    all_finite = bool(raw_df["finite"].all())
    delta_loss_lattice = float(delta_df["delta_loss_lattice"].mean())
    delta_loss_random = float(delta_df["delta_loss_random"].mean())

    if not all_finite:
        conclusion = "inconclusive"
    elif abs(delta_loss_lattice - delta_loss_random) < 1e-6:
        conclusion = "no meaningful difference"
    elif 0 < delta_loss_lattice < delta_loss_random:
        conclusion = "generic structure helps"
    elif delta_loss_lattice >= delta_loss_random:
        conclusion = "connectome-specific"
    else:
        conclusion = "inconclusive"

    summary_map = {row["model_kind"]: row for row in summary_df.to_dict(orient="records")}
    boot = {row["metric"]: row for row in bootstrap_df.to_dict(orient="records")}

    lines = [
        "# Stage 3 Connectome vs Lattice Matched Steps",
        "",
        "Scope:",
        "- Matched-step comparison.",
        "- Same canonical Stage 3 setup.",
        "- Same seeds [0,1,2].",
        "- Fixed 5 iterations.",
        "- 3 graph types: connectome, random, ring-lattice.",
        "- No model/loss changes.",
        "",
        "Graph table:",
        "",
        "| Model | Nodes | Edges | Self-loops | Free params | Fixed params |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model_kind in MODEL_KINDS:
        row = model_table[model_kind]
        lines.append(
            f"| {model_kind} | {row['nodes']} | {row['edges']} | {row['self_loops']} | {row['free_parameters']} | {row['fixed_parameters']} |"
        )

    lines.extend(
        [
            "",
            "Degree diagnostics:",
            "",
            "| Model | In-degree mean/std | Out-degree mean/std |",
            "|---|---:|---:|",
            f"| connectome | {connectome_stats['in_mean']:.4f} / {connectome_stats['in_std']:.4f} | {connectome_stats['out_mean']:.4f} / {connectome_stats['out_std']:.4f} |",
            f"| random | {random_stats['in_mean']:.4f} / {random_stats['in_std']:.4f} | {random_stats['out_mean']:.4f} / {random_stats['out_std']:.4f} |",
            f"| ringlattice | {ring_stats['in_mean']:.4f} / {ring_stats['in_std']:.4f} | {ring_stats['out_mean']:.4f} / {ring_stats['out_std']:.4f} |",
            "",
            "Summary at iter 5:",
            "",
            "| Model | Loss | Activity | Elapsed time |",
            "|---|---:|---:|---:|",
        ]
    )
    for model_kind in MODEL_KINDS:
        row = summary_map[model_kind]
        lines.append(
            f"| {model_kind} | {fmt(row['loss_mean'], row['loss_std'], 3)} | {fmt(row['activity_mean'], row['activity_std'], 3)} | {fmt(row['elapsed_mean'], row['elapsed_std'], 3)} |"
        )

    lines.extend(
        [
            "",
            "Delta summaries vs connectome:",
            "",
            f"- random loss/activity/time: {delta_df['delta_loss_random'].mean():.4f} +/- {delta_df['delta_loss_random'].std(ddof=1):.4f} / {delta_df['delta_activity_random'].mean():.4f} +/- {delta_df['delta_activity_random'].std(ddof=1):.4f} / {delta_df['delta_time_random'].mean():.4f} +/- {delta_df['delta_time_random'].std(ddof=1):.4f}",
            f"- lattice loss/activity/time: {delta_df['delta_loss_lattice'].mean():.4f} +/- {delta_df['delta_loss_lattice'].std(ddof=1):.4f} / {delta_df['delta_activity_lattice'].mean():.4f} +/- {delta_df['delta_activity_lattice'].std(ddof=1):.4f} / {delta_df['delta_time_lattice'].mean():.4f} +/- {delta_df['delta_time_lattice'].std(ddof=1):.4f}",
            "",
            "Bootstrap 95% CI:",
            "",
            f"- delta_loss_random: mean={boot['delta_loss_random']['mean']:.4f}, CI=[{boot['delta_loss_random']['ci_low']:.4f}, {boot['delta_loss_random']['ci_high']:.4f}]",
            f"- delta_loss_lattice: mean={boot['delta_loss_lattice']['mean']:.4f}, CI=[{boot['delta_loss_lattice']['ci_low']:.4f}, {boot['delta_loss_lattice']['ci_high']:.4f}]",
            f"- delta_activity_random: mean={boot['delta_activity_random']['mean']:.4f}, CI=[{boot['delta_activity_random']['ci_low']:.4f}, {boot['delta_activity_random']['ci_high']:.4f}]",
            f"- delta_activity_lattice: mean={boot['delta_activity_lattice']['mean']:.4f}, CI=[{boot['delta_activity_lattice']['ci_low']:.4f}, {boot['delta_activity_lattice']['ci_high']:.4f}]",
            "",
            "Stability notes:",
            f"- all finite: {all_finite}",
            "",
            "Interpretation:",
            f"- Conclusion label: {conclusion}",
            "- Optional 10-step extension recommendation: reasonable only if you want to test whether the lattice stays finite and whether its gap to connectome narrows or persists under the exact same pipeline.",
        ]
    )

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(lines))

    print(summary_df.to_string(index=False))
    print(delta_df.to_string(index=False))
    print(bootstrap_df.to_string(index=False))
    print(f"Conclusion: {conclusion}")


if __name__ == "__main__":
    main()
