"""Benchmarking package for optimizer comparison."""

from __future__ import annotations

__version__ = "3.0.0"

from .adapters import BaseAdapter, create_adapter
from .harness import BenchmarkRunner, run_single
from .problems import describe_problems, get_problem, get_problem_names
from .profiling import ProfilingResult, run_py_spy_if_available, run_suggestion_throughput
from .ray_runner import run_ray_experiment, run_ray_local
from .report import build_report
from .suite import DEFAULT_BACKENDS, DEFAULT_OPTIMIZERS, run_suite

__all__ = [
    "__version__",
    "BaseAdapter",
    "create_adapter",
    "describe_problems",
    "get_problem",
    "get_problem_names",
    "BenchmarkRunner",
    "run_single",
    "run_ray_local",
    "run_ray_experiment",
    "ProfilingResult",
    "run_suggestion_throughput",
    "run_py_spy_if_available",
    "build_report",
    "run_suite",
    "DEFAULT_BACKENDS",
    "DEFAULT_OPTIMIZERS",
]
