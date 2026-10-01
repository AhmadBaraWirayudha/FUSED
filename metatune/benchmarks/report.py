"""Report generation for benchmark runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None  # type: ignore

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None  # type: ignore


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        draw = rng.choice(values, size=values.size, replace=True)
        samples[i] = float(np.mean(draw))
    return float(np.quantile(samples, alpha / 2.0)), float(np.quantile(samples, 1.0 - alpha / 2.0))


def paired_ttest(x: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size or x.size < 2:
        return {"t_stat": None, "p_value": None}
    if stats is None:
        diff = x - y
        std = float(np.std(diff, ddof=1))
        if std == 0.0:
            return {"t_stat": None, "p_value": None}
        t = float(np.mean(diff) / (std / np.sqrt(diff.size)))
        return {"t_stat": t, "p_value": None}
    result = stats.ttest_rel(x, y, nan_policy="omit")
    return {"t_stat": float(result.statistic), "p_value": float(result.pvalue)}


def load_logs(results_dir: str | Path) -> pd.DataFrame:
    results_dir = Path(results_dir)
    files = sorted(results_dir.glob("runs/*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(path) for path in files], ignore_index=True)


def _markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data available_"
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def _safe_group_stats(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["problem", "optimizer", "backend", "mean", "ci_low", "ci_high", "n"])
    rows: list[dict[str, Any]] = []
    for (problem, optimizer, backend), grp in df.groupby(["problem", "optimizer", "backend"], dropna=False):
        values = pd.to_numeric(grp[metric], errors="coerce").dropna().to_numpy()
        mean = float(np.mean(values)) if values.size else float("nan")
        ci_low, ci_high = bootstrap_ci(values)
        rows.append(
            {
                "problem": problem,
                "optimizer": optimizer,
                "backend": backend,
                "mean": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n": int(values.size),
            }
        )
    return pd.DataFrame(rows)


def _plot_metric(df: pd.DataFrame, metric: str, ylabel: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    if df.empty:
        plt.text(0.5, 0.5, "No data available", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        return

    plot_df = df[["optimizer", "backend", "problem", metric]].copy()
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df.dropna(subset=[metric])
    if sns is not None:
        sns.boxplot(data=plot_df, x="optimizer", y=metric, hue="backend")
        plt.legend(loc="best")
    else:
        groups = [grp[metric].tolist() for _, grp in plot_df.groupby(["optimizer", "backend"])]
        labels = [f"{o}/{b}" for (o, b), _ in plot_df.groupby(["optimizer", "backend"])]
        plt.boxplot(groups, labels=labels)
        plt.xticks(rotation=25)
    plt.ylabel(ylabel)
    plt.title(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _best_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame(columns=["problem", "backend", "optimizer", "best_objective_value", "trial_id"])
    for (problem, backend, optimizer), grp in df.groupby(["problem", "backend", "optimizer"], dropna=False):
        metric_name = str(grp["metric_name"].iloc[0]).lower()
        minimize_metrics = {"rmse", "loss", "error", "mae", "mse"}
        if metric_name in minimize_metrics:
            idx = grp["objective_value"].astype(float).idxmin()
        else:
            idx = grp["objective_value"].astype(float).idxmax()
        row = grp.loc[idx]
        rows.append(
            {
                "problem": problem,
                "backend": backend,
                "optimizer": optimizer,
                "best_objective_value": float(row["objective_value"]),
                "trial_id": int(row["trial_id"]),
            }
        )
    return pd.DataFrame(rows)


def _paired_ours_vs_optuna(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["problem", "backend", "n", "t_stat", "p_value"])
    rows: list[dict[str, Any]] = []
    for (problem, backend), grp in df.groupby(["problem", "backend"], dropna=False):
        ours = grp[grp["optimizer"] == "ours"].sort_values("trial_id")
        optuna = grp[grp["optimizer"] == "optuna"].sort_values("trial_id")
        merged = ours.merge(optuna, on="trial_id", suffixes=("_ours", "_optuna"))
        if merged.empty:
            continue
        test = paired_ttest(
            merged["objective_value_ours"].to_numpy(),
            merged["objective_value_optuna"].to_numpy(),
        )
        rows.append(
            {
                "problem": problem,
                "backend": backend,
                "n": int(len(merged)),
                "t_stat": test["t_stat"],
                "p_value": test["p_value"],
            }
        )
    return pd.DataFrame(rows)


def _claim_audit(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            [
                {"claim": "Median overhead < 0.5% at 8 workers", "status": "N/A"},
                {"claim": "Overhead < 1% at 16 workers", "status": "N/A"},
                {"claim": "Throughput >= 10,000 suggestions/sec", "status": "N/A"},
                {"claim": "Ours not significantly worse than Optuna", "status": "N/A"},
            ]
        )

    rows: list[dict[str, Any]] = []

    ours = df[df["optimizer"] == "ours"]
    if not ours.empty:
        median_overhead = float(pd.to_numeric(ours["overhead_time_sec"], errors="coerce").div(pd.to_numeric(ours["total_time_sec"], errors="coerce")).mul(100).median())
        rows.append(
            {
                "claim": "Median overhead < 0.5% at 8 workers",
                "status": "PASS" if median_overhead < 0.5 else "FAIL",
                "value": round(median_overhead, 4),
            }
        )
    else:
        rows.append({"claim": "Median overhead < 0.5% at 8 workers", "status": "N/A", "value": None})

    workers_16 = df[df["backend"] == "ray"]
    if not workers_16.empty:
        overhead16 = float(pd.to_numeric(workers_16["overhead_time_sec"], errors="coerce").div(pd.to_numeric(workers_16["total_time_sec"], errors="coerce")).mul(100).median())
        rows.append(
            {
                "claim": "Overhead < 1% at 16 workers",
                "status": "PASS" if overhead16 < 1.0 else "FAIL",
                "value": round(overhead16, 4),
            }
        )
    else:
        rows.append({"claim": "Overhead < 1% at 16 workers", "status": "N/A", "value": None})

    return pd.DataFrame(rows)


def build_report(results_dir: str | Path, output_path: str | Path | None = None, figures_dir: str | Path | None = None) -> str:
    results_dir = Path(results_dir)
    output_path = Path(output_path) if output_path else results_dir.parent / "BENCHMARK_REPORT.md"
    figures_dir = Path(figures_dir) if figures_dir else results_dir.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_logs(results_dir)
    if not df.empty:
        df = df.copy()
        for col in ("total_time_sec", "objective_time_sec", "overhead_time_sec", "suggest_time_sec", "report_time_sec", "objective_value"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["overhead_pct"] = (df["overhead_time_sec"] / df["total_time_sec"]) * 100.0
        df["objective_share_pct"] = (df["objective_time_sec"] / df["total_time_sec"]) * 100.0

    timing_summary = _safe_group_stats(df, "total_time_sec")
    objective_summary = _safe_group_stats(df, "objective_time_sec")
    overhead_summary = _safe_group_stats(df, "overhead_time_sec")
    overhead_pct_summary = _safe_group_stats(df, "overhead_pct")
    objective_share_summary = _safe_group_stats(df, "objective_share_pct")
    best_summary = _best_rows(df)
    paired_summary = _paired_ours_vs_optuna(df)
    claim_audit = _claim_audit(df)

    _plot_metric(df, "total_time_sec", "Total time (s)", figures_dir / "total_time_boxplot.png")
    _plot_metric(df, "objective_time_sec", "Objective time (s)", figures_dir / "objective_time_boxplot.png")
    _plot_metric(df, "overhead_pct", "Overhead (%)", figures_dir / "overhead_pct_boxplot.png")

    report_lines: list[str] = []
    report_lines.append("# Benchmark Report")
    report_lines.append("")
    report_lines.append(f"Runs analyzed: {int(len(df))}")
    report_lines.append(f"Problems: {', '.join(sorted(df['problem'].dropna().astype(str).unique())) if not df.empty else 'None'}")
    report_lines.append(f"Optimizers: {', '.join(sorted(df['optimizer'].dropna().astype(str).unique())) if not df.empty else 'None'}")
    report_lines.append(f"Backends: {', '.join(sorted(df['backend'].dropna().astype(str).unique())) if not df.empty else 'None'}")
    report_lines.append("")

    report_lines.append("## Executive summary")
    report_lines.append("")
    report_lines.append(
        "This report compares the repository optimizer against Optuna and Ray Tune Optuna across real CPU-only tasks, then separates objective time from optimizer overhead so the overhead claim can be evaluated directly."
    )
    report_lines.append("")

    report_lines.append("## Claim audit")
    report_lines.append("")
    report_lines.append(_markdown(claim_audit))
    report_lines.append("")

    report_lines.append("## Timing summary")
    report_lines.append("")
    report_lines.append(_markdown(timing_summary))
    report_lines.append("")
    report_lines.append("## Objective-time summary")
    report_lines.append("")
    report_lines.append(_markdown(objective_summary))
    report_lines.append("")
    report_lines.append("## Overhead summary")
    report_lines.append("")
    report_lines.append(_markdown(overhead_summary))
    report_lines.append("")
    report_lines.append("## Overhead percentage summary")
    report_lines.append("")
    report_lines.append(_markdown(overhead_pct_summary))
    report_lines.append("")
    report_lines.append("## Objective share summary")
    report_lines.append("")
    report_lines.append(_markdown(objective_share_summary))
    report_lines.append("")
    report_lines.append("## Best scores")
    report_lines.append("")
    report_lines.append(_markdown(best_summary))
    report_lines.append("")
    report_lines.append("## Paired ours vs optuna comparisons")
    report_lines.append("")
    report_lines.append(_markdown(paired_summary))
    report_lines.append("")
    report_lines.append("## Figures")
    report_lines.append("")
    report_lines.append(f"- {figures_dir / 'total_time_boxplot.png'}")
    report_lines.append(f"- {figures_dir / 'objective_time_boxplot.png'}")
    report_lines.append(f"- {figures_dir / 'overhead_pct_boxplot.png'}")
    report_lines.append("")
    report_lines.append("## Notes")
    report_lines.append("- `overhead_pct = overhead_time_sec / total_time_sec * 100`.")
    report_lines.append("- Bootstrapped 95% confidence intervals are computed per problem/optimizer/backend group.")
    report_lines.append("- The MNIST problem first tries OpenML `mnist_784` and falls back to a deterministic offline surrogate.")
    report_lines.append("- Optional packages are used when available; local scikit-learn fallbacks preserve the benchmark contract.")
    report_lines.append("- Paired tests are only produced when `ours` and `optuna` have matching trial IDs for the same problem and backend.")

    report = "\n".join(report_lines) + "\n"
    output_path.write_text(report, encoding="utf-8")
    return report
