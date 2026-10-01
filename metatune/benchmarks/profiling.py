"""Profiling helpers for optimizer suggestion throughput and memory."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Mapping

import psutil

from .adapters import BaseAdapter


@dataclass
class ProfilingResult:
    optimizer: str
    problem: str
    n_calls: int
    total_seconds: float
    suggestions_per_second: float
    memory_before_mb: float
    memory_after_mb: float
    memory_delta_mb: float
    py_spy_available: bool
    py_spy_status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def run_suggestion_throughput(
    adapter: BaseAdapter,
    search_space: Mapping[str, Any],
    n_calls: int = 10_000,
    *,
    output_dir: str | Path | None = None,
) -> ProfilingResult:
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)

    started = time.perf_counter()
    for trial_id in range(n_calls):
        config = adapter.suggest(trial_id)
        adapter.report(trial_id, 0.0)
        _ = config
    total = time.perf_counter() - started

    mem_after = process.memory_info().rss / (1024 * 1024)
    result = ProfilingResult(
        optimizer=adapter.name,
        problem="throughput",
        n_calls=n_calls,
        total_seconds=float(total),
        suggestions_per_second=float(n_calls / total) if total > 0 else float("inf"),
        memory_before_mb=float(mem_before),
        memory_after_mb=float(mem_after),
        memory_delta_mb=float(mem_after - mem_before),
        py_spy_available=shutil.which("py-spy") is not None,
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "throughput.json").write_text(result.to_json(), encoding="utf-8")

    return result


def run_py_spy_if_available(output_dir: str | Path, command: list[str]) -> str | None:
    if shutil.which("py-spy") is None:
        return "py-spy not installed; skipped"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outfile = output_dir / "py_spy.svg"
    cmd = ["py-spy", "record", "--format", "svg", "--output", str(outfile), "--", *command]
    try:
        completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except Exception as exc:
        return f"py-spy failed: {exc}"

    if completed.returncode != 0:
        stderr = (completed.stderr or completed.stdout or "").strip()
        return f"py-spy exited {completed.returncode}: {stderr}"
    return str(outfile)
