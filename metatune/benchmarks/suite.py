"""Suite orchestration for one-command benchmark execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .harness import BenchmarkRunner
from .problems import get_problem_names
from .report import build_report


DEFAULT_OPTIMIZERS = ("ours", "optuna", "raytune")
DEFAULT_BACKENDS = ("serial", "processes", "ray")


def run_suite(
    *,
    problems: Iterable[str] | None = None,
    optimizers: Iterable[str] | None = None,
    backends: Iterable[str] | None = None,
    trials: int = 10,
    workers: int | None = None,
    seed: int = 42,
    output_dir: str | Path | None = None,
    build_final_report: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir or Path(__file__).resolve().parent / "results")
    output_dir.mkdir(parents=True, exist_ok=True)

    problems = list(problems or get_problem_names())
    optimizers = list(optimizers or DEFAULT_OPTIMIZERS)
    backends = list(backends or DEFAULT_BACKENDS)

    summaries: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []

    for problem in problems:
        for optimizer in optimizers:
            for backend in backends:
                runner = BenchmarkRunner(
                    problem_name=problem,
                    optimizer_name=optimizer,
                    n_trials=trials,
                    parallel_backend=backend,
                    workers=workers,
                    seed=seed,
                    output_dir=output_dir,
                )
                df, summary = runner.run()
                summaries.append(summary)
                frames.append(df)

    report_text = None
    report_path = output_dir.parent / "BENCHMARK_REPORT.md"
    if build_final_report:
        report_text = build_report(output_dir, output_path=report_path, figures_dir=output_dir.parent / "figures")

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return {
        "runs": summaries,
        "rows": int(len(combined)),
        "report_path": str(report_path),
        "report_generated": bool(report_text is not None),
    }
