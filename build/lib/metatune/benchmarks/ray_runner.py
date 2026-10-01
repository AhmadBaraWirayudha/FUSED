"""Ray-local execution helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .harness import BenchmarkRunner


def run_ray_local(
    problem_name: str,
    optimizer_name: str,
    n_trials: int,
    *,
    workers: int | None = None,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    runner = BenchmarkRunner(
        problem_name=problem_name,
        optimizer_name=optimizer_name,
        n_trials=n_trials,
        parallel_backend="ray",
        workers=workers,
        seed=seed,
        output_dir=output_dir,
    )
    return runner.run()


def run_ray_experiment(
    problem_name: str,
    optimizers: list[str],
    n_trials: int,
    *,
    workers: int | None = None,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for optimizer in optimizers:
        _, summary = run_ray_local(
            problem_name=problem_name,
            optimizer_name=optimizer,
            n_trials=n_trials,
            workers=workers,
            seed=seed,
            output_dir=output_dir,
        )
        rows.append(summary)
    return rows
