"""CLI entry point for benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import __version__
from .adapters import create_adapter
from .harness import BenchmarkRunner
from .problems import describe_problems, get_problem, get_problem_names
from .profiling import run_suggestion_throughput
from .ray_runner import run_ray_experiment, run_ray_local
from .report import build_report
from .suite import DEFAULT_BACKENDS, DEFAULT_OPTIMIZERS, run_suite


def _load_config(path: Path) -> dict:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to read config.yaml") from exc
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _warn_python_version() -> None:
    if sys.version_info[:2] != (3, 11):
        print(
            f"Warning: Python 3.11 expected; running on {sys.version_info.major}.{sys.version_info.minor}.",
            file=sys.stderr,
        )


def _default_trials(config: dict, problem: str) -> int:
    defaults = (config or {}).get("defaults", {})
    return int(defaults.get(problem, 10))


def _default_workers(config: dict) -> int:
    runtime = (config or {}).get("runtime", {})
    workers = runtime.get("default_workers")
    if workers is not None:
        return int(workers)
    return BenchmarkRunner.default_workers()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m metatune.benchmarks.cli", description="Optimizer benchmark suite")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")), help="Path to config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run one benchmark")
    run.add_argument("--problem", required=True, help="Problem name or alias")
    run.add_argument("--optimizers", nargs="+", default=list(DEFAULT_OPTIMIZERS), help="Optimizers to run")
    run.add_argument("--trials", type=int, default=None, help="Trial budget")
    run.add_argument("--backend", choices=["serial", "processes", "ray"], default="serial")
    run.add_argument("--workers", type=int, default=None)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--output-dir", default=str(Path(__file__).with_name("results")))

    run_ray = sub.add_parser("run-ray", help="Run a Ray-local benchmark")
    run_ray.add_argument("--problem", required=True)
    run_ray.add_argument("--optimizers", nargs="+", default=list(DEFAULT_OPTIMIZERS))
    run_ray.add_argument("--trials", type=int, default=None)
    run_ray.add_argument("--workers", type=int, default=None)
    run_ray.add_argument("--seed", type=int, default=42)
    run_ray.add_argument("--output-dir", default=str(Path(__file__).with_name("results")))

    profile = sub.add_parser("profile", help="Profile optimizer suggestion throughput")
    profile.add_argument("--problem", required=True)
    profile.add_argument("--optimizer", default="ours", choices=list(DEFAULT_OPTIMIZERS))
    profile.add_argument("--calls", type=int, default=10_000)
    profile.add_argument("--seed", type=int, default=42)
    profile.add_argument("--output-dir", default=str(Path(__file__).with_name("results")))

    report = sub.add_parser("report", help="Generate the Markdown report")
    report.add_argument("--results-dir", default=str(Path(__file__).with_name("results")))
    report.add_argument("--output", default=None)
    report.add_argument("--figures-dir", default=None)

    suite = sub.add_parser("suite", help="Run the full benchmark suite and generate a report")
    suite.add_argument("--problems", nargs="+", default=None)
    suite.add_argument("--optimizers", nargs="+", default=list(DEFAULT_OPTIMIZERS))
    suite.add_argument("--backends", nargs="+", default=list(DEFAULT_BACKENDS))
    suite.add_argument("--trials", type=int, default=None)
    suite.add_argument("--workers", type=int, default=None)
    suite.add_argument("--seed", type=int, default=42)
    suite.add_argument("--output-dir", default=str(Path(__file__).with_name("results")))

    sub.add_parser("list-problems", help="List available benchmark problems")
    return parser


def main(argv: list[str] | None = None) -> int:
    _warn_python_version()
    parser = build_parser()
    args = parser.parse_args(argv)
    config = _load_config(Path(args.config))

    if args.command == "list-problems":
        rows = describe_problems()
        print(json.dumps(rows, indent=2))
        return 0

    output_dir = Path(getattr(args, "output_dir", Path(__file__).with_name("results")))
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "run":
        workers = int(args.workers or _default_workers(config))
        trials = int(args.trials or _default_trials(config, args.problem))
        for optimizer in args.optimizers:
            summary = BenchmarkRunner(
                problem_name=args.problem,
                optimizer_name=optimizer,
                n_trials=trials,
                parallel_backend=args.backend,
                workers=workers,
                seed=args.seed,
                output_dir=output_dir,
            ).run()[1]
            print(json.dumps(summary, indent=2, default=str))
        return 0

    if args.command == "run-ray":
        workers = int(args.workers or _default_workers(config))
        trials = int(args.trials or _default_trials(config, args.problem))
        for optimizer in args.optimizers:
            _, summary = run_ray_local(
                problem_name=args.problem,
                optimizer_name=optimizer,
                n_trials=trials,
                workers=workers,
                seed=args.seed,
                output_dir=output_dir,
            )
            print(json.dumps(summary, indent=2, default=str))
        return 0

    if args.command == "profile":
        problem = get_problem(args.problem)
        adapter = create_adapter(args.optimizer, problem.search_space, problem.direction, args.seed)
        result = run_suggestion_throughput(adapter, problem.search_space, n_calls=args.calls, output_dir=output_dir)
        print(result.to_json())
        print("py-spy:", "available" if __import__("shutil").which("py-spy") else "not installed; skipped")
        return 0

    if args.command == "report":
        report_path = Path(args.output) if args.output else None
        figures_dir = Path(args.figures_dir) if args.figures_dir else None
        report_text = build_report(args.results_dir, output_path=report_path, figures_dir=figures_dir)
        print(report_text)
        return 0

    if args.command == "suite":
        workers = int(args.workers or _default_workers(config))
        trials = int(args.trials or 25)
        summary = run_suite(
            problems=args.problems or get_problem_names(),
            optimizers=args.optimizers,
            backends=args.backends,
            trials=trials,
            workers=workers,
            seed=args.seed,
            output_dir=output_dir,
            build_final_report=True,
        )
        print(json.dumps(summary, indent=2, default=str))
        return 0

    parser.error("Unknown command")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
