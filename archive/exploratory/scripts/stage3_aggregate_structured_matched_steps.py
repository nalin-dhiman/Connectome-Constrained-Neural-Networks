#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


STEP_HORIZONS = (5, 10)
MODEL_KINDS = ("connectome", "random", "smallworld")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate connectome vs structured matched-step results."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("results/stage3_connectome_vs_structured_matched_steps"),
    )
    parser.add_argument(
        "--aggregated-csv",
        type=Path,
        default=Path("results/stage3_connectome_vs_structured_matched_steps/aggregated_metrics.csv"),
    )
    parser.add_argument(
        "--delta-csv",
        type=Path,
        default=Path("results/stage3_connectome_vs_structured_matched_steps/delta_metrics.csv"),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/stage3_connectome_vs_structured.md"),
    )
    return parser.parse_args()


def fmt(mean: float, std: float, n: int) -> str:
    if n == 0 or np.isnan(mean):
        return "n/a"
    if n == 1 or np.isnan(std):
        return f"{mean:.4f} (n=1)"
    return f"{mean:.4f} +/- {std:.4f}"


def seed_dirs(step_root: Path) -> list[Path]:
    dirs = sorted(path for path in step_root.glob("seed_*") if path.is_dir())
    if not dirs:
        raise FileNotFoundError(f"No seed directories found under {step_root}")
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


def main() -> None:
    args = parse_args()
    for output_path in (args.aggregated_csv, args.delta_csv, args.report_path):
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")

    aggregated_rows: list[dict] = []
    delta_rows: list[dict] = []
    model_table = None

    for step in STEP_HORIZONS:
        step_root = args.input_root / f"steps_{step}"
        for seed_dir in seed_dirs(step_root):
            seed = int(seed_dir.name.split("_")[-1])
            summary = json.loads((seed_dir / "summary.json").read_text())
            if model_table is None:
                model_table = {
                    model_kind: {
                        "nodes": int(summary[model_kind]["n_nodes"]),
                        "edges": int(summary[model_kind]["n_edges"]),
                        "self_loops": int(summary[model_kind]["self_loops"]),
                        "params": (
                            int(summary[model_kind]["free_parameters"]),
                            int(summary[model_kind]["fixed_parameters"]),
                        ),
                    }
                    for model_kind in MODEL_KINDS
                }

            curves = {model_kind: load_curve(seed_dir, model_kind) for model_kind in MODEL_KINDS}
            for model_kind in MODEL_KINDS:
                curve = curves[model_kind]
                aggregated_rows.append(
                    {
                        "steps": step,
                        "seed": seed,
                        "model_kind": model_kind,
                        "loss": value_at_iteration(curve, step, "task_loss"),
                        "activity": value_at_iteration(curve, step, "activity_abs_mean"),
                        "elapsed": value_at_iteration(curve, step, "elapsed_sec"),
                        "finite": bool(curve["finite"].all()),
                    }
                )

            connectome = curves["connectome"]
            random = curves["random"]
            smallworld = curves["smallworld"]
            delta_rows.append(
                {
                    "steps": step,
                    "seed": seed,
                    "delta_loss_random": value_at_iteration(random, step, "task_loss")
                    - value_at_iteration(connectome, step, "task_loss"),
                    "delta_loss_smallworld": value_at_iteration(smallworld, step, "task_loss")
                    - value_at_iteration(connectome, step, "task_loss"),
                    "delta_activity_random": value_at_iteration(random, step, "activity_abs_mean")
                    - value_at_iteration(connectome, step, "activity_abs_mean"),
                    "delta_activity_smallworld": value_at_iteration(smallworld, step, "activity_abs_mean")
                    - value_at_iteration(connectome, step, "activity_abs_mean"),
                    "delta_time_random": value_at_iteration(random, step, "elapsed_sec")
                    - value_at_iteration(connectome, step, "elapsed_sec"),
                    "delta_time_smallworld": value_at_iteration(smallworld, step, "elapsed_sec")
                    - value_at_iteration(connectome, step, "elapsed_sec"),
                }
            )

    raw_df = pd.DataFrame(aggregated_rows).sort_values(["steps", "model_kind", "seed"]).reset_index(drop=True)
    delta_df = pd.DataFrame(delta_rows).sort_values(["steps", "seed"]).reset_index(drop=True)

    summary_rows: list[dict] = []
    for step in STEP_HORIZONS:
        step_df = raw_df.loc[raw_df["steps"].eq(step)]
        for model_kind in MODEL_KINDS:
            model_df = step_df.loc[step_df["model_kind"].eq(model_kind)]
            summary_rows.append(
                {
                    "steps": step,
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
    args.aggregated_csv.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(args.input_root / "raw_metrics.csv", index=False)
    summary_df.to_csv(args.aggregated_csv, index=False)
    delta_df.to_csv(args.delta_csv, index=False)

    summary_map = {
        (row["steps"], row["model_kind"]): row
        for row in summary_df.to_dict(orient="records")
    }
    delta_map = {
        step: delta_df.loc[delta_df["steps"].eq(step)]
        for step in STEP_HORIZONS
    }

    step5 = delta_map[5]
    step10 = delta_map[10]
    loss_smallworld_10 = float(step10["delta_loss_smallworld"].mean())
    loss_random_10 = float(step10["delta_loss_random"].mean())
    if loss_smallworld_10 < loss_random_10 and loss_smallworld_10 > 0:
        conclusion = "connectome-specific"
    elif abs(loss_smallworld_10 - loss_random_10) < 1e-6:
        conclusion = "no difference"
    elif loss_smallworld_10 <= 0:
        conclusion = "structure helps"
    else:
        conclusion = "inconclusive"

    lines = [
        "# Stage 3 Connectome vs Structured",
        "",
        "Scope:",
        "- Matched-step comparison.",
        "- Same setup.",
        "- Three graph types only.",
        "- No other changes.",
        "",
        "Model table:",
        "",
        "| Model | Nodes | Edges | Self-loops | Params |",
        "|---|---:|---:|---:|---:|",
    ]
    for model_kind in MODEL_KINDS:
        params = model_table[model_kind]["params"]
        lines.append(
            f"| {model_kind} | {model_table[model_kind]['nodes']} | {model_table[model_kind]['edges']} | {model_table[model_kind]['self_loops']} | {params[0] + params[1]} |"
        )

    for step in STEP_HORIZONS:
        lines.extend(
            [
                "",
                f"Metrics at iter {step}:",
                "",
                "| Model | Loss | Activity | Time |",
                "|---|---:|---:|---:|",
            ]
        )
        for model_kind in MODEL_KINDS:
            row = summary_map[(step, model_kind)]
            lines.append(
                f"| {model_kind} | {fmt(row['loss_mean'], row['loss_std'], 3)} | {fmt(row['activity_mean'], row['activity_std'], 3)} | {fmt(row['elapsed_mean'], row['elapsed_std'], 3)} |"
            )
        lines.extend(
            [
                "",
                f"Delta summary vs connectome at iter {step}:",
                f"- random loss/activity/time: {delta_map[step]['delta_loss_random'].mean():.4f} / {delta_map[step]['delta_activity_random'].mean():.4f} / {delta_map[step]['delta_time_random'].mean():.4f}",
                f"- smallworld loss/activity/time: {delta_map[step]['delta_loss_smallworld'].mean():.4f} / {delta_map[step]['delta_activity_smallworld'].mean():.4f} / {delta_map[step]['delta_time_smallworld'].mean():.4f}",
            ]
        )

    lines.extend(
        [
            "",
            "Interpretation:",
            f"- Conclusion label: {conclusion}",
        ]
    )
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(lines))

    print(summary_df.to_string(index=False))
    print(delta_df.to_string(index=False))
    print(f"Conclusion: {conclusion}")


if __name__ == "__main__":
    main()
