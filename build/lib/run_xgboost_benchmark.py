"""One-touch runner for the XGBoost benchmark problems: the real diabetes_xgboost,
the make_regression-based synthetic_xgboost, and 4 methods that generate synthetic
data to resemble the real diabetes data's own joint distribution
(synthetic_xgboost_gaussian/gmm/kde/bootstrap) -- ours vs Optuna, 30 trials each,
then a report. Meant to be launched via run_xgboost_benchmark.bat, but works
standalone too:

    python run_xgboost_benchmark.py [--trials 30] [--seed 42] [--problems ...]
"""
from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# import name -> what to tell pip if it's missing (only used for the message; the actual
# install always goes through the project's own `benchmarks` extra so versions stay pinned
# to what metatune/benchmarks/requirements.txt actually declares)
REQUIRED_MODULES = ["numpy", "scipy", "pandas", "sklearn", "xgboost", "optuna", "matplotlib", "seaborn", "psutil", "yaml"]


def _missing_modules() -> list[str]:
    missing = []
    for mod in REQUIRED_MODULES:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(mod)
    return missing


def _ensure_dependencies() -> None:
    missing = _missing_modules()
    if not missing:
        return
    print(f"Installing benchmark dependencies (missing: {', '.join(missing)}) ...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", f"{REPO_ROOT}[fast,benchmarks]"],
        check=True,
    )
    still_missing = _missing_modules()
    if still_missing:
        print(f"Still missing after install attempt: {', '.join(still_missing)}")
        print("Install manually with: pip install .[fast,benchmarks]")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--problems",
        nargs="+",
        default=[
            "diabetes_xgboost",
            "synthetic_xgboost",
            "synthetic_xgboost_gaussian",
            "synthetic_xgboost_gmm",
            "synthetic_xgboost_kde",
            "synthetic_xgboost_bootstrap",
        ],
        help="Which XGBoost problems to run. Default: the real diabetes data, the "
        "make_regression synthetic data, and all 4 real-data-like synthetic methods.",
    )
    args = parser.parse_args()

    _ensure_dependencies()
    warnings.filterwarnings("ignore")

    sys.path.insert(0, str(REPO_ROOT))
    from sklearn.datasets import load_diabetes

    from metatune.benchmarks.harness import BenchmarkRunner
    from metatune.benchmarks.report import build_report
    from metatune.benchmarks.synthetic_data import compare_distributions, generate_like

    real_X, real_y = load_diabetes(return_X_y=True)
    print("How closely does each synthetic method match the real diabetes data?")
    print(f"{'method':<12}{'mean_abs_diff':>16}{'std_abs_diff':>16}{'corr_frobenius_diff':>22}")
    for method in ("gaussian", "gmm", "kde", "bootstrap"):
        synth_X, synth_y = generate_like(method, real_X, real_y, n_samples=len(real_X), seed=args.seed)
        stats = compare_distributions(real_X, real_y, synth_X, synth_y)
        print(
            f"{method:<12}{stats['mean_abs_diff_avg']:>16.4f}"
            f"{stats['std_abs_diff_avg']:>16.4f}{stats['correlation_frobenius_diff']:>22.4f}"
        )
    print("(lower is closer to the real data on every column; 0 would be identical)\n")

    output_dir = REPO_ROOT / "metatune" / "benchmarks" / "results" / "xgboost_synth_run"

    print(f"Running ours vs Optuna on {args.problems}, {args.trials} trials each, seed {args.seed}\n")
    for problem in args.problems:
        for optimizer in ("ours", "optuna"):
            print(f"  {problem} / {optimizer} ...", end=" ", flush=True)
            _, summary = BenchmarkRunner(
                problem_name=problem,
                optimizer_name=optimizer,
                n_trials=args.trials,
                parallel_backend="serial",
                seed=args.seed,
                output_dir=str(output_dir),
            ).run()
            print(
                f"best={summary['best_value']:.4f}  "
                f"mean_objective_time={summary['mean_objective_time_sec']:.3f}s  "
                f"overhead={summary['mean_overhead_pct']:.2f}%"
            )

    report_output_path = REPO_ROOT / "XGBOOST_BENCHMARK_REPORT.md"
    build_report(output_dir, output_path=report_output_path)
    print(f"\nReport written to: {report_output_path}")


if __name__ == "__main__":
    main()
