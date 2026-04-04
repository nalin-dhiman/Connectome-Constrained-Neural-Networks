#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ITERATION_POINTS = (1, 3, 5, 10)
MODEL_KINDS = ("connectome", "random")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate matched-step connectome-vs-random 10-step results."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("results/main_results/stage3_connectome_vs_random_matched_steps_10"),
    )
    parser.add_argument(
        "--aggregated-csv",
        type=Path,
        default=Path("results/main_results/stage3_connectome_vs_random_matched_steps_10/aggregated_metrics.csv"),
    )
    parser.add_argument(
        "--delta-csv",
        type=Path,
        default=Path("results/main_results/stage3_connectome_vs_random_matched_steps_10/delta_metrics.csv"),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("docs/generated/stage3_connectome_vs_random_matched_steps_10.md"),
    )
    return parser.parse_args()


def seed_dirs(input_root: Path) -> list[Path]:
    dirs = sorted(path for path in input_root.glob("seed_*") if path.is_dir())
    if not dirs:
        raise FileNotFoundError(f"No seed directories found under {input_root}")
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


def fmt(mean: float, std: float, n: int) -> str:
    if n == 0 or np.isnan(mean):
        return "n/a"
    if n == 1 or np.isnan(std):
        return f"{mean:.4f} (n=1)"
    return f"{mean:.4f} +/- {std:.4f}"


def main() -> None:
    args = parse_args()
    for output_path in (args.aggregated_csv, args.delta_csv, args.report_path):
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")

    raw_rows: list[dict] = []
    delta_rows: list[dict] = []

    for seed_dir in seed_dirs(args.input_root):
        seed = int(seed_dir.name.split("_")[-1])
        curves = {model_kind: load_curve(seed_dir, model_kind) for model_kind in MODEL_KINDS}

        for model_kind in MODEL_KINDS:
            curve = curves[model_kind]
            row = {"seed": seed, "model_kind": model_kind, "all_finite": bool(curve["finite"].all())}
            for iteration in ITERATION_POINTS:
                row[f"loss_iter_{iteration}"] = value_at_iteration(curve, iteration, "task_loss")
                row[f"activity_iter_{iteration}"] = value_at_iteration(curve, iteration, "activity_abs_mean")
                row[f"elapsed_iter_{iteration}"] = value_at_iteration(curve, iteration, "elapsed_sec")
            raw_rows.append(row)

        connectome_curve = curves["connectome"]
        random_curve = curves["random"]
        delta_rows.append(
            {
                "seed": seed,
                "delta_loss_iter10": value_at_iteration(random_curve, 10, "task_loss")
                - value_at_iteration(connectome_curve, 10, "task_loss"),
                "delta_activity_iter10": value_at_iteration(random_curve, 10, "activity_abs_mean")
                - value_at_iteration(connectome_curve, 10, "activity_abs_mean"),
                "delta_time_iter10": value_at_iteration(random_curve, 10, "elapsed_sec")
                - value_at_iteration(connectome_curve, 10, "elapsed_sec"),
            }
        )

    raw_df = pd.DataFrame(raw_rows).sort_values(["model_kind", "seed"]).reset_index(drop=True)
    delta_df = pd.DataFrame(delta_rows).sort_values("seed").reset_index(drop=True)

    aggregated_rows: list[dict] = []
    for model_kind in MODEL_KINDS:
        model_df = raw_df.loc[raw_df["model_kind"].eq(model_kind)].reset_index(drop=True)
        row = {"model_kind": model_kind, "all_finite": bool(model_df["all_finite"].all())}
        for prefix in ("loss", "activity", "elapsed"):
            for iteration in ITERATION_POINTS:
                column = f"{prefix}_iter_{iteration}"
                values = model_df[column].to_numpy(dtype=float)
                row[f"{column}_mean"] = float(np.nanmean(values))
                row[f"{column}_std"] = float(np.nanstd(values, ddof=1))
                row[f"{column}_n"] = int(np.sum(~np.isnan(values)))
        aggregated_rows.append(row)

    aggregated_df = pd.DataFrame(aggregated_rows)
    args.aggregated_csv.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(args.input_root / "raw_metrics.csv", index=False)
    aggregated_df.to_csv(args.aggregated_csv, index=False)
    delta_df.to_csv(args.delta_csv, index=False)

    agg = {row["model_kind"]: row for row in aggregated_df.to_dict(orient="records")}
    all_finite = bool(raw_df["all_finite"].all())

    if not all_finite:
        conclusion = "unstable"
    else:
        loss10 = float(delta_df["delta_loss_iter10"].mean())
        activity10 = float(delta_df["delta_activity_iter10"].mean())
        if loss10 > 0 and activity10 > 0:
            conclusion = "persists"
        elif loss10 > 0 or activity10 > 0:
            conclusion = "weakens"
        else:
            conclusion = "disappears"

    report_lines = [
        "# Stage 3 Connectome vs Random Matched Steps 10",
        "",
        "Scope:",
        "- Matched-step extension.",
        "- Same setup.",
        "- Only iteration horizon increased.",
        "",
        "Table A: loss",
        "",
        "| Model | Iter 1 | Iter 3 | Iter 5 | Iter 10 |",
        "|---|---:|---:|---:|---:|",
        f"| connectome | {fmt(agg['connectome']['loss_iter_1_mean'], agg['connectome']['loss_iter_1_std'], agg['connectome']['loss_iter_1_n'])} | {fmt(agg['connectome']['loss_iter_3_mean'], agg['connectome']['loss_iter_3_std'], agg['connectome']['loss_iter_3_n'])} | {fmt(agg['connectome']['loss_iter_5_mean'], agg['connectome']['loss_iter_5_std'], agg['connectome']['loss_iter_5_n'])} | {fmt(agg['connectome']['loss_iter_10_mean'], agg['connectome']['loss_iter_10_std'], agg['connectome']['loss_iter_10_n'])} |",
        f"| random | {fmt(agg['random']['loss_iter_1_mean'], agg['random']['loss_iter_1_std'], agg['random']['loss_iter_1_n'])} | {fmt(agg['random']['loss_iter_3_mean'], agg['random']['loss_iter_3_std'], agg['random']['loss_iter_3_n'])} | {fmt(agg['random']['loss_iter_5_mean'], agg['random']['loss_iter_5_std'], agg['random']['loss_iter_5_n'])} | {fmt(agg['random']['loss_iter_10_mean'], agg['random']['loss_iter_10_std'], agg['random']['loss_iter_10_n'])} |",
        "",
        "Table B: activity",
        "",
        "| Model | Iter 1 | Iter 3 | Iter 5 | Iter 10 |",
        "|---|---:|---:|---:|---:|",
        f"| connectome | {fmt(agg['connectome']['activity_iter_1_mean'], agg['connectome']['activity_iter_1_std'], agg['connectome']['activity_iter_1_n'])} | {fmt(agg['connectome']['activity_iter_3_mean'], agg['connectome']['activity_iter_3_std'], agg['connectome']['activity_iter_3_n'])} | {fmt(agg['connectome']['activity_iter_5_mean'], agg['connectome']['activity_iter_5_std'], agg['connectome']['activity_iter_5_n'])} | {fmt(agg['connectome']['activity_iter_10_mean'], agg['connectome']['activity_iter_10_std'], agg['connectome']['activity_iter_10_n'])} |",
        f"| random | {fmt(agg['random']['activity_iter_1_mean'], agg['random']['activity_iter_1_std'], agg['random']['activity_iter_1_n'])} | {fmt(agg['random']['activity_iter_3_mean'], agg['random']['activity_iter_3_std'], agg['random']['activity_iter_3_n'])} | {fmt(agg['random']['activity_iter_5_mean'], agg['random']['activity_iter_5_std'], agg['random']['activity_iter_5_n'])} | {fmt(agg['random']['activity_iter_10_mean'], agg['random']['activity_iter_10_std'], agg['random']['activity_iter_10_n'])} |",
        "",
        "Table C: time",
        "",
        "| Model | Time at iter 10 |",
        "|---|---:|",
        f"| connectome | {fmt(agg['connectome']['elapsed_iter_10_mean'], agg['connectome']['elapsed_iter_10_std'], agg['connectome']['elapsed_iter_10_n'])} |",
        f"| random | {fmt(agg['random']['elapsed_iter_10_mean'], agg['random']['elapsed_iter_10_std'], agg['random']['elapsed_iter_10_n'])} |",
        "",
        "Delta summary at iter 10:",
        "",
        f"- delta_loss_iter10: {delta_df['delta_loss_iter10'].mean():.4f} +/- {delta_df['delta_loss_iter10'].std(ddof=1):.4f}",
        f"- delta_activity_iter10: {delta_df['delta_activity_iter10'].mean():.4f} +/- {delta_df['delta_activity_iter10'].std(ddof=1):.4f}",
        f"- delta_time_iter10: {delta_df['delta_time_iter10'].mean():.4f} +/- {delta_df['delta_time_iter10'].std(ddof=1):.4f}",
        "",
        "Stability notes:",
        f"- all runs finite: {all_finite}",
        f"- conclusion label: {conclusion}",
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report_lines))

    print(aggregated_df.to_string(index=False))
    print(delta_df.to_string(index=False))
    print(f"Conclusion: {conclusion}")


if __name__ == "__main__":
    main()
