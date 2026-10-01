"""Benchmark harness for serial, process, and Ray execution modes."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
import json
import multiprocessing as mp
import numpy as np
from pathlib import Path
import os
import time
from typing import Any, Mapping
from uuid import uuid4

import pandas as pd

from .adapters import BaseAdapter, create_adapter
from .problems import ProblemDefinition, get_problem
from .types import ParamSpec


@dataclass
class TrialLog:
    trial_id: int
    problem: str
    optimizer: str
    backend: str
    config_json: str
    metric_name: str
    objective_value: float
    total_time_sec: float
    objective_time_sec: float
    overhead_time_sec: float
    suggest_time_sec: float
    report_time_sec: float
    worker_pid: int | None = None
    timestamp_utc: str | None = None


def _extract_metric(objective_result: Mapping[str, Any], problem: ProblemDefinition) -> float:
    if problem.metric_name in objective_result:
        return float(objective_result[problem.metric_name])
    if "loss" in objective_result:
        loss = float(objective_result["loss"])
        return -loss if problem.direction == "maximize" else loss
    for value in objective_result.values():
        if isinstance(value, (int, float)):
            return float(value)
    raise ValueError(f"Objective did not return a usable metric for {problem.name}")


@lru_cache(maxsize=64)
def _cached_objective(problem_name: str, seed: int):
    return get_problem(problem_name).make_objective(seed)


def _evaluate_objective(config: dict[str, Any], problem_name: str, seed: int) -> tuple[dict[str, Any], float, int]:
    objective = _cached_objective(problem_name, seed)
    started = time.perf_counter()
    result = objective(config)
    elapsed = time.perf_counter() - started
    return dict(result), float(elapsed), os.getpid()


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_adapter(
    adapter: BaseAdapter | None,
    optimizer_name: str | None,
    search_space: dict[str, ParamSpec],
    direction: str,
    seed: int,
) -> BaseAdapter:
    if adapter is not None:
        return adapter
    if optimizer_name is None:
        raise ValueError("Either adapter or optimizer_name must be provided")
    return create_adapter(optimizer_name, search_space, direction, seed)


class BenchmarkRunner:
    """Run one problem/optimizer pair under a selected backend."""

    def __init__(
        self,
        problem_name: str,
        optimizer_name: str | None = None,
        n_trials: int = 10,
        parallel_backend: str = "serial",
        workers: int | None = None,
        seed: int = 42,
        output_dir: str | Path | None = None,
        adapter: BaseAdapter | None = None,
    ) -> None:
        self.problem: ProblemDefinition = get_problem(problem_name)
        self.optimizer_name = optimizer_name.lower() if optimizer_name else (adapter.name if adapter is not None else "ours")
        self.n_trials = int(n_trials)
        self.parallel_backend = parallel_backend.lower()
        self.workers = int(workers or self.default_workers())
        self.seed = int(seed)
        self.adapter = adapter
        self.output_dir = Path(output_dir or Path(__file__).resolve().parent / "results")
        self.runs_dir = self.output_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def default_workers() -> int:
        try:
            import psutil
            cpu = psutil.cpu_count(logical=True)
            return max(1, int(cpu or (os.cpu_count() or 1)))
        except Exception:
            return max(1, int(os.cpu_count() or 1))

    def _trial_log(
        self,
        trial_id: int,
        config: dict[str, Any],
        objective_value: float,
        objective_time: float,
        suggest_time: float,
        report_time: float,
        total_time: float,
        worker_pid: int | None,
    ) -> TrialLog:
        overhead = max(0.0, total_time - objective_time)
        return TrialLog(
            trial_id=trial_id,
            problem=self.problem.name,
            optimizer=self.optimizer_name,
            backend=self.parallel_backend,
            config_json=json.dumps(config, sort_keys=True, default=str),
            metric_name=self.problem.metric_name,
            objective_value=float(objective_value),
            total_time_sec=float(total_time),
            objective_time_sec=float(objective_time),
            overhead_time_sec=float(overhead),
            suggest_time_sec=float(suggest_time),
            report_time_sec=float(report_time),
            worker_pid=worker_pid,
            timestamp_utc=_now_utc(),
        )

    def _run_serial(self, adapter: BaseAdapter) -> list[TrialLog]:
        logs: list[TrialLog] = []
        objective = self.problem.make_objective(self.seed)
        for trial_id in range(self.n_trials):
            t0 = time.perf_counter()
            s0 = time.perf_counter()
            config = adapter.suggest(trial_id)
            suggest_time = time.perf_counter() - s0

            o0 = time.perf_counter()
            result = objective(config)
            objective_time = time.perf_counter() - o0
            metric_value = _extract_metric(result, self.problem)

            r0 = time.perf_counter()
            adapter.report(trial_id, metric_value)
            report_time = time.perf_counter() - r0

            total_time = time.perf_counter() - t0
            logs.append(
                self._trial_log(
                    trial_id,
                    config,
                    metric_value,
                    objective_time,
                    suggest_time,
                    report_time,
                    total_time,
                    os.getpid(),
                )
            )
        return logs

    def _run_parallel_processes(self, adapter: BaseAdapter) -> list[TrialLog]:
        logs: list[TrialLog] = []
        pending: dict[Any, tuple[int, dict[str, Any], float, float]] = {}
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=self.workers, mp_context=ctx) as executor:
            next_trial = 0
            while next_trial < self.n_trials and len(pending) < self.workers:
                s0 = time.perf_counter()
                config = adapter.suggest(next_trial)
                suggest_time = time.perf_counter() - s0
                submitted = time.perf_counter()
                future = executor.submit(_evaluate_objective, config, self.problem.name, self.seed)
                pending[future] = (next_trial, config, suggest_time, submitted)
                next_trial += 1

            while pending:
                done, _ = wait(set(pending.keys()), return_when=FIRST_COMPLETED)
                for future in done:
                    trial_id, config, suggest_time, submitted = pending.pop(future)
                    result, objective_time, worker_pid = future.result()
                    metric_value = _extract_metric(result, self.problem)

                    r0 = time.perf_counter()
                    adapter.report(trial_id, metric_value)
                    report_time = time.perf_counter() - r0

                    total_time = time.perf_counter() - submitted
                    logs.append(
                        self._trial_log(
                            trial_id,
                            config,
                            metric_value,
                            objective_time,
                            suggest_time,
                            report_time,
                            total_time,
                            worker_pid,
                        )
                    )

                    if next_trial < self.n_trials:
                        s1 = time.perf_counter()
                        next_config = adapter.suggest(next_trial)
                        next_suggest_time = time.perf_counter() - s1
                        next_submitted = time.perf_counter()
                        future2 = executor.submit(_evaluate_objective, next_config, self.problem.name, self.seed)
                        pending[future2] = (next_trial, next_config, next_suggest_time, next_submitted)
                        next_trial += 1

        logs.sort(key=lambda row: row.trial_id)
        return logs

    def _run_ray(self, adapter: BaseAdapter, workers: int | None = None) -> list[TrialLog]:
        try:
            import ray
        except Exception as exc:
            raise RuntimeError("Ray is not installed") from exc

        workers = int(workers or self.workers)
        ray.shutdown()
        ray.init(num_cpus=workers, ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)

        @ray.remote
        def _evaluate(config: dict[str, Any], problem_name: str, seed: int):
            return _evaluate_objective(config, problem_name, seed)

        logs: list[TrialLog] = []
        pending: dict[Any, tuple[int, dict[str, Any], float, float]] = {}
        next_trial = 0

        while next_trial < self.n_trials and len(pending) < workers:
            s0 = time.perf_counter()
            config = adapter.suggest(next_trial)
            suggest_time = time.perf_counter() - s0
            submitted = time.perf_counter()
            future = _evaluate.remote(config, self.problem.name, self.seed)
            pending[future] = (next_trial, config, suggest_time, submitted)
            next_trial += 1

        try:
            while pending:
                done, _ = ray.wait(list(pending.keys()), num_returns=1)
                future = done[0]
                trial_id, config, suggest_time, submitted = pending.pop(future)
                result, objective_time, worker_pid = ray.get(future)
                metric_value = _extract_metric(result, self.problem)

                r0 = time.perf_counter()
                adapter.report(trial_id, metric_value)
                report_time = time.perf_counter() - r0

                total_time = time.perf_counter() - submitted
                logs.append(
                    self._trial_log(
                        trial_id,
                        config,
                        metric_value,
                        objective_time,
                        suggest_time,
                        report_time,
                        total_time,
                        worker_pid,
                    )
                )
                if next_trial < self.n_trials:
                    s1 = time.perf_counter()
                    next_config = adapter.suggest(next_trial)
                    next_suggest_time = time.perf_counter() - s1
                    next_submitted = time.perf_counter()
                    future2 = _evaluate.remote(next_config, self.problem.name, self.seed)
                    pending[future2] = (next_trial, next_config, next_suggest_time, next_submitted)
                    next_trial += 1
        finally:
            ray.shutdown()

        logs.sort(key=lambda row: row.trial_id)
        return logs

    def _build_summary(self, df: pd.DataFrame, adapter: BaseAdapter, run_id: str, csv_path: Path, summary_path: Path) -> dict[str, Any]:
        best = adapter.final_result()
        mean_total = float(df["total_time_sec"].mean()) if not df.empty else float("nan")
        mean_objective = float(df["objective_time_sec"].mean()) if not df.empty else float("nan")
        mean_overhead = float(df["overhead_time_sec"].mean()) if not df.empty else float("nan")
        mean_overhead_pct = float(pd.to_numeric(df["overhead_time_sec"], errors="coerce").div(pd.to_numeric(df["total_time_sec"], errors="coerce")).replace([np.inf, -np.inf], np.nan).mean() * 100.0) if not df.empty else float("nan")
        objective_share = float(pd.to_numeric(df["objective_time_sec"], errors="coerce").div(pd.to_numeric(df["total_time_sec"], errors="coerce")).replace([np.inf, -np.inf], np.nan).mean() * 100.0) if not df.empty else float("nan")
        summary = {
            "run_id": run_id,
            "csv_path": str(csv_path),
            "summary_path": str(summary_path),
            "problem": self.problem.name,
            "optimizer": self.optimizer_name,
            "backend": self.parallel_backend,
            "workers": self.workers,
            "trials": int(len(df)),
            "best_value": best.get("best_value"),
            "best_config": best.get("best_config"),
            "best_trial_id": best.get("best_trial_id"),
            "mean_total_time_sec": mean_total,
            "mean_objective_time_sec": mean_objective,
            "mean_overhead_time_sec": mean_overhead,
            "mean_overhead_pct": mean_overhead_pct,
            "mean_objective_share_pct": objective_share,
        }
        return summary

    def run(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        adapter = _make_adapter(self.adapter, self.optimizer_name, self.problem.search_space, self.problem.direction, self.seed)
        if self.parallel_backend == "serial":
            logs = self._run_serial(adapter)
        elif self.parallel_backend == "processes":
            logs = self._run_parallel_processes(adapter)
        elif self.parallel_backend == "ray":
            logs = self._run_ray(adapter, workers=self.workers)
        else:
            raise ValueError(f"Unknown backend: {self.parallel_backend}")

        df = pd.DataFrame([asdict(row) for row in logs])
        run_id = f"{self.problem.name}_{self.optimizer_name}_{self.parallel_backend}_{uuid4().hex}"
        csv_path = self.runs_dir / f"{run_id}.csv"
        df.to_csv(csv_path, index=False)

        summary_path = self.runs_dir / f"{run_id}.json"
        summary = self._build_summary(df, adapter, run_id, csv_path, summary_path)
        summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        return df, summary


def run_single(
    problem_name: str,
    optimizer_name: str,
    n_trials: int,
    parallel_backend: str = "serial",
    workers: int | None = None,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    runner = BenchmarkRunner(
        problem_name=problem_name,
        optimizer_name=optimizer_name,
        n_trials=n_trials,
        parallel_backend=parallel_backend,
        workers=workers,
        seed=seed,
        output_dir=output_dir,
    )
    df, summary = runner.run()
    summary["rows"] = len(df)
    return summary
