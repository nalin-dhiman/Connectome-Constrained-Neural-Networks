#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ITERATION_POINTS = (1, 3, 5)
MODEL_KINDS = ("connectome", "random")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate matched-step connectome-vs-random results."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("results/main_results/stage3_connectome_vs_random_matched_steps"),
    )
    parser.add_argument(
        "--aggregated-csv",
        type=Path,
        default=Path("results/main_results/stage3_connectome_vs_random_matched_steps/aggregated_metrics.csv"),
    )
    parser.add_argument(
        "--delta-csv",
        type=Path,
        default=Path("results/main_results/stage3_connectome_vs_random_matched_steps/delta_metrics.csv"),
    )
    parser.add_argument(
        "--bootstrap-csv",
        type=Path,
        default=Path("results/main_results/stage3_connectome_vs_random_matched_steps/bootstrap_iter5.csv"),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("docs/generated/stage3_connectome_vs_random_matched_steps.md"),
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    return parser.parse_args()


def load_curve(seed_dir: Path, model_kind: str) -> pd.DataFrame:
    path = seed_dir / f"{model_kind}_curve.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing curve file: {path}")
    return pd.read_csv(path)


def load_summary(seed_dir: Path) -> dict:
    path = seed_dir / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file: {path}")
    return pd.read_json(path, typ="series").to_dict()


def seed_dirs(input_root: Path) -> list[Path]:
    dirs = sorted(path for path in input_root.glob("seed_*") if path.is_dir())
    if not dirs:
        raise FileNotFoundError(f"No seed directories found under {input_root}")
    return dirs


def value_at_iteration(curve_df: pd.DataFrame, iteration: int, column: str) -> float:
    row = curve_df.loc[curve_df["iteration"].eq(iteration)]
    if row.empty:
        return float("nan")
    return float(row.iloc[0][column])


def bootstrap_ci(values: np.ndarray, resamples: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(0)
    n = len(values)
    bootstrap_means = np.empty(resamples, dtype=float)
    for idx in range(resamples):
        sample = rng.choice(values, size=n, replace=True)
        bootstrap_means[idx] = float(np.mean(sample))
    return (
        float(np.mean(values)),
        float(np.percentile(bootstrap_means, 2.5)),
        float(np.percentile(bootstrap_means, 97.5)),
    )


def fmt(mean: float, std: float, n: int) -> str:
    if n == 0 or np.isnan(mean):
        return "n/a"
    if n == 1 or np.isnan(std):
        return f"{mean:.4f} (n=1)"
    return f"{mean:.4f} +/- {std:.4f}"


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
            raw_row = {"seed": seed, "model_kind": model_kind}
            for iteration in ITERATION_POINTS:
                raw_row[f"loss_iter_{iteration}"] = value_at_iteration(
                    curve, iteration, "task_loss"
                )
                raw_row[f"activity_iter_{iteration}"] = value_at_iteration(
                    curve, iteration, "activity_abs_mean"
                )
                raw_row[f"elapsed_iter_{iteration}"] = value_at_iteration(
                    curve, iteration, "elapsed_sec"
                )
            raw_row["all_finite"] = bool(curve["finite"].all())
            raw_rows.append(raw_row)

        connectome_curve = curves["connectome"]
        random_curve = curves["random"]
        delta_rows.append(
            {
                "seed": seed,
                "delta_loss_iter5": value_at_iteration(random_curve, 5, "task_loss")
                - value_at_iteration(connectome_curve, 5, "task_loss"),
                "delta_activity_iter5": value_at_iteration(
                    random_curve, 5, "activity_abs_mean"
                )
                - value_at_iteration(connectome_curve, 5, "activity_abs_mean"),
                "delta_time_iter5": value_at_iteration(random_curve, 5, "elapsed_sec")
                - value_at_iteration(connectome_curve, 5, "elapsed_sec"),
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

    bootstrap_rows = []
    for column in ("delta_loss_iter5", "delta_activity_iter5", "delta_time_iter5"):
        values = delta_df[column].to_numpy(dtype=float)
        mean, ci_low, ci_high = bootstrap_ci(values, args.bootstrap_resamples)
        bootstrap_rows.append(
            {
                "metric": column,
                "mean": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    bootstrap_df = pd.DataFrame(bootstrap_rows)

    args.aggregated_csv.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(args.input_root / "raw_metrics.csv", index=False)
    aggregated_df.to_csv(args.aggregated_csv, index=False)
    delta_df.to_csv(args.delta_csv, index=False)
    bootstrap_df.to_csv(args.bootstrap_csv, index=False)

    agg = {row["model_kind"]: row for row in aggregated_df.to_dict(orient="records")}
    boot = {row["metric"]: row for row in bootstrap_df.to_dict(orient="records")}

    all_finite = bool(raw_df["all_finite"].all())
    loss_delta_mean = float(delta_df["delta_loss_iter5"].mean())
    activity_delta_mean = float(delta_df["delta_activity_iter5"].mean())

    if not all_finite:
        conclusion = "inconclusive"
    elif loss_delta_mean > 0 and activity_delta_mean > 0:
        conclusion = "optimization advantage"
    elif loss_delta_mean <= 0 and activity_delta_mean > 0:
        conclusion = "activation advantage only"
    elif loss_delta_mean <= 0 and activity_delta_mean <= 0:
        conclusion = "runtime advantage only"
    else:
        conclusion = "unstable"

    report_lines = [
        "# Stage 3 Connectome vs Random Matched Steps",
        "",
        "Scope:",
        "- Matched-step comparison, not a benchmark.",
        "- Same canonical Stage 3 setup.",
        "- Same seeds [0, 1, 2].",
        "- Fixed 5 iterations for both models.",
        "- No wall-clock stopping.",
        "",
        "Model table:",
        "",
        "| Model | Nodes | Edges | Self-loops | Free params | Fixed params |",
        "|---|---:|---:|---:|---:|---:|",
        "| connectome | 45669 | 1513231 | 12380 | 734 | 2959 |",
        "| random | 45669 | 1513231 | 12380 | 734 | 2959 |",
        "",
        "Table A: loss",
        "",
        "| Model | Iter 1 | Iter 3 | Iter 5 |",
        "|---|---:|---:|---:|",
        f"| connectome | {fmt(agg['connectome']['loss_iter_1_mean'], agg['connectome']['loss_iter_1_std'], agg['connectome']['loss_iter_1_n'])} | {fmt(agg['connectome']['loss_iter_3_mean'], agg['connectome']['loss_iter_3_std'], agg['connectome']['loss_iter_3_n'])} | {fmt(agg['connectome']['loss_iter_5_mean'], agg['connectome']['loss_iter_5_std'], agg['connectome']['loss_iter_5_n'])} |",
        f"| random | {fmt(agg['random']['loss_iter_1_mean'], agg['random']['loss_iter_1_std'], agg['random']['loss_iter_1_n'])} | {fmt(agg['random']['loss_iter_3_mean'], agg['random']['loss_iter_3_std'], agg['random']['loss_iter_3_n'])} | {fmt(agg['random']['loss_iter_5_mean'], agg['random']['loss_iter_5_std'], agg['random']['loss_iter_5_n'])} |",
        "",
        "Table B: activity",
        "",
        "| Model | Iter 1 | Iter 3 | Iter 5 |",
        "|---|---:|---:|---:|",
        f"| connectome | {fmt(agg['connectome']['activity_iter_1_mean'], agg['connectome']['activity_iter_1_std'], agg['connectome']['activity_iter_1_n'])} | {fmt(agg['connectome']['activity_iter_3_mean'], agg['connectome']['activity_iter_3_std'], agg['connectome']['activity_iter_3_n'])} | {fmt(agg['connectome']['activity_iter_5_mean'], agg['connectome']['activity_iter_5_std'], agg['connectome']['activity_iter_5_n'])} |",
        f"| random | {fmt(agg['random']['activity_iter_1_mean'], agg['random']['activity_iter_1_std'], agg['random']['activity_iter_1_n'])} | {fmt(agg['random']['activity_iter_3_mean'], agg['random']['activity_iter_3_std'], agg['random']['activity_iter_3_n'])} | {fmt(agg['random']['activity_iter_5_mean'], agg['random']['activity_iter_5_std'], agg['random']['activity_iter_5_n'])} |",
        "",
        "Table C: elapsed time",
        "",
        "| Model | Time at iter 5 |",
        "|---|---:|",
        f"| connectome | {fmt(agg['connectome']['elapsed_iter_5_mean'], agg['connectome']['elapsed_iter_5_std'], agg['connectome']['elapsed_iter_5_n'])} |",
        f"| random | {fmt(agg['random']['elapsed_iter_5_mean'], agg['random']['elapsed_iter_5_std'], agg['random']['elapsed_iter_5_n'])} |",
        "",
        "Delta summary (random - connectome at iter 5):",
        "",
        f"- loss: {delta_df['delta_loss_iter5'].mean():.4f} +/- {delta_df['delta_loss_iter5'].std(ddof=1):.4f}",
        f"- activity: {delta_df['delta_activity_iter5'].mean():.4f} +/- {delta_df['delta_activity_iter5'].std(ddof=1):.4f}",
        f"- elapsed time: {delta_df['delta_time_iter5'].mean():.4f} +/- {delta_df['delta_time_iter5'].std(ddof=1):.4f}",
        "",
        "Bootstrap 95% CI for iter-5 deltas:",
        "",
        f"- loss: mean={boot['delta_loss_iter5']['mean']:.4f}, CI=[{boot['delta_loss_iter5']['ci_low']:.4f}, {boot['delta_loss_iter5']['ci_high']:.4f}]",
        f"- activity: mean={boot['delta_activity_iter5']['mean']:.4f}, CI=[{boot['delta_activity_iter5']['ci_low']:.4f}, {boot['delta_activity_iter5']['ci_high']:.4f}]",
        f"- elapsed time: mean={boot['delta_time_iter5']['mean']:.4f}, CI=[{boot['delta_time_iter5']['ci_low']:.4f}, {boot['delta_time_iter5']['ci_high']:.4f}]",
        "",
        "Stability notes:",
        f"- all runs finite: {all_finite}",
        f"- conclusion label: {conclusion}",
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report_lines))

    print(aggregated_df.to_string(index=False))
    print(delta_df.to_string(index=False))
    print(bootstrap_df.to_string(index=False))
    print(f"Conclusion: {conclusion}")


if __name__ == "__main__":
    main()
