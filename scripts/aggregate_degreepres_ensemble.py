#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from revision_control_utils import bootstrap_mean_ci, format_mean_std, load_curve, write_markdown


DEFAULT_INPUT_ROOT = Path("results/revision_results/revision_degpres_ensemble/steps_5")
DEFAULT_REPORT_PATH = Path("docs/generated/revision_degpres_ensemble.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate degree-preserving ensemble step-5 results."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args()


def value_at_iteration(curve_df: pd.DataFrame, iteration: int, column: str) -> float:
    row = curve_df.loc[curve_df["iteration"].eq(iteration)]
    if row.empty:
        return float("nan")
    return float(row.iloc[0][column])


def stats(values: np.ndarray) -> tuple[float, float, int]:
    values = np.asarray(values, dtype=float)
    valid = values[np.isfinite(values)]
    if len(valid) == 0:
        return float("nan"), float("nan"), 0
    mean = float(np.mean(valid))
    std = float(np.std(valid, ddof=1)) if len(valid) > 1 else float("nan")
    return mean, std, int(len(valid))


def main() -> None:
    args = parse_args()
    if args.report_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {args.report_path}")
    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root not found: {args.input_root}")

    combo_rows = []
    for sample_dir in sorted(path for path in args.input_root.glob("sample_*") if path.is_dir()):
        sample_id = int(sample_dir.name.split("_")[-1])
        for seed_dir in sorted(path for path in sample_dir.glob("seed_*") if path.is_dir()):
            seed = int(seed_dir.name.split("_")[-1])
            connectome_curve = load_curve(seed_dir, "connectome_curve.csv")
            degree_curve = load_curve(seed_dir, "degreepres_curve.csv")
            connectome_loss = value_at_iteration(connectome_curve, 5, "task_loss")
            connectome_activity = value_at_iteration(connectome_curve, 5, "activity_abs_mean")
            connectome_elapsed = value_at_iteration(connectome_curve, 5, "elapsed_sec")
            degree_loss = value_at_iteration(degree_curve, 5, "task_loss")
            degree_activity = value_at_iteration(degree_curve, 5, "activity_abs_mean")
            degree_elapsed = value_at_iteration(degree_curve, 5, "elapsed_sec")
            combo_rows.append(
                {
                    "sample_id": sample_id,
                    "seed": seed,
                    "connectome_loss": connectome_loss,
                    "connectome_activity": connectome_activity,
                    "connectome_elapsed": connectome_elapsed,
                    "degreepres_loss": degree_loss,
                    "degreepres_activity": degree_activity,
                    "degreepres_elapsed": degree_elapsed,
                    "delta_loss": degree_loss - connectome_loss,
                    "delta_activity": degree_activity - connectome_activity,
                    "delta_elapsed": degree_elapsed - connectome_elapsed,
                    "connectome_finite": bool(connectome_curve["finite"].all()),
                    "degreepres_finite": bool(degree_curve["finite"].all()),
                }
            )

    if not combo_rows:
        raise RuntimeError(f"No sample-seed results found under {args.input_root}")

    combo_df = pd.DataFrame(combo_rows).sort_values(["sample_id", "seed"])
    sample_df = (
        combo_df.groupby("sample_id", as_index=False)[
            [
                "connectome_loss",
                "connectome_activity",
                "connectome_elapsed",
                "degreepres_loss",
                "degreepres_activity",
                "degreepres_elapsed",
                "delta_loss",
                "delta_activity",
                "delta_elapsed",
            ]
        ]
        .mean()
        .sort_values("sample_id")
    )

    aggregated_path = args.input_root / "aggregated_metrics.csv"
    sample_level_path = args.input_root / "sample_level_metrics.csv"
    delta_path = args.input_root / "delta_metrics.csv"
    combo_df.to_csv(aggregated_path, index=False)
    sample_df.to_csv(sample_level_path, index=False)
    combo_df[["sample_id", "seed", "delta_loss", "delta_activity", "delta_elapsed"]].to_csv(
        delta_path, index=False
    )

    connectome_loss = stats(combo_df["connectome_loss"].to_numpy())
    degree_loss = stats(combo_df["degreepres_loss"].to_numpy())
    connectome_activity = stats(combo_df["connectome_activity"].to_numpy())
    degree_activity = stats(combo_df["degreepres_activity"].to_numpy())
    connectome_elapsed = stats(combo_df["connectome_elapsed"].to_numpy())
    degree_elapsed = stats(combo_df["degreepres_elapsed"].to_numpy())
    delta_loss_ci = bootstrap_mean_ci(combo_df["delta_loss"].to_numpy())
    delta_activity_ci = bootstrap_mean_ci(combo_df["delta_activity"].to_numpy())
    delta_elapsed_ci = bootstrap_mean_ci(combo_df["delta_elapsed"].to_numpy())

    lines = [
        "# Degree-Preserving Ensemble Revision",
        "",
        "This report aggregates the 5-step degree-preserving ensemble over all "
        "sample-seed combinations. Connectome curves are reused from the completed "
        "fair-initialization Phase 2 runs because the connectome branch is invariant "
        "across degree-preserving samples.",
        "",
        f"- Degree-preserving samples: {combo_df['sample_id'].nunique()}",
        f"- Seeds per sample: {combo_df['seed'].nunique()}",
        f"- Total sample-seed combinations: {len(combo_df)}",
        f"- All runs finite: {bool(combo_df['connectome_finite'].all() and combo_df['degreepres_finite'].all())}",
        "",
        "| Metric @ iter 5 | Connectome | Degree-preserving |",
        "|---|---:|---:|",
        f"| Loss | {format_mean_std(*connectome_loss)} | {format_mean_std(*degree_loss)} |",
        f"| Activity | {format_mean_std(*connectome_activity)} | {format_mean_std(*degree_activity)} |",
        f"| Elapsed (s) | {format_mean_std(*connectome_elapsed)} | {format_mean_std(*degree_elapsed)} |",
        "",
        f"- delta_loss (degreepres - connectome): mean={delta_loss_ci[0]:.4f}, 95% CI=[{delta_loss_ci[1]:.4f}, {delta_loss_ci[2]:.4f}]",
        f"- delta_activity (degreepres - connectome): mean={delta_activity_ci[0]:.4f}, 95% CI=[{delta_activity_ci[1]:.4f}, {delta_activity_ci[2]:.4f}]",
        f"- delta_elapsed (degreepres - connectome): mean={delta_elapsed_ci[0]:.4f}, 95% CI=[{delta_elapsed_ci[1]:.4f}, {delta_elapsed_ci[2]:.4f}]",
        "",
        "## Interpretation",
    ]

    if delta_activity_ci[0] <= 0.0 and abs(delta_loss_ci[0]) < 0.01:
        lines.append(
            "Across multiple degree-preserving rewired samples, the connectome does not recover "
            "a robust activity or loss advantage. This strengthens the corrected paper claim that "
            "the original effect was sensitive to null-model design."
        )
    elif delta_activity_ci[1] > 0.0 or delta_loss_ci[1] > 0.0:
        lines.append(
            "The ensemble leaves some positive connectome-versus-degree-preserving signal, so the "
            "corrected claim would need to be stated more cautiously."
        )
    else:
        lines.append(
            "The ensemble remains mixed; no strong topology-favoring interpretation is supported."
        )

    write_markdown(args.report_path, lines)
    print(f"Wrote aggregated metrics to {aggregated_path}")
    print(f"Wrote sample-level metrics to {sample_level_path}")
    print(f"Wrote delta metrics to {delta_path}")
    print(f"Wrote report to {args.report_path}")


if __name__ == "__main__":
    main()
