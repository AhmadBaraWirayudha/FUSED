from pathlib import Path

import pytest

from metatune.benchmarks.adapters import create_adapter
from metatune.benchmarks.harness import BenchmarkRunner
from metatune.benchmarks.problems import get_problem, get_problem_names
from metatune.benchmarks.suite import run_suite


def test_problem_registry():
    names = get_problem_names()
    assert "digits_svm" in names
    assert "breast_cancer_lightgbm" in names
    problem = get_problem("svm")
    assert problem.metric_name == "accuracy"
    assert len(problem.search_space) >= 3


def test_small_serial_run(tmp_path: Path):
    runner = BenchmarkRunner(
        problem_name="svm",
        optimizer_name="ours",
        n_trials=2,
        parallel_backend="serial",
        workers=1,
        seed=42,
        output_dir=tmp_path,
    )
    df, summary = runner.run()
    assert len(df) == 2
    assert summary["trials"] == 2
    assert (tmp_path / "runs").exists()


def test_synthetic_xgboost_problem_registered():
    names = get_problem_names()
    assert "synthetic_xgboost" in names
    problem = get_problem("synth")  # alias
    assert problem.name == "synthetic_xgboost"
    assert problem.metric_name == "rmse"
    assert problem.direction == "minimize"


@pytest.mark.parametrize("method", ["gaussian", "gmm", "kde", "bootstrap"])
def test_realistic_synthetic_xgboost_variants_registered(method):
    names = get_problem_names()
    full_name = f"synthetic_xgboost_{method}"
    assert full_name in names
    problem = get_problem(f"synth_{method}")  # alias
    assert problem.name == full_name
    assert problem.metric_name == "rmse"
    obj = problem.make_objective(42)
    config = {k: spec.default for k, spec in problem.search_space.items()}
    result = obj(config)
    assert result["rmse"] > 0


def test_synthetic_xgboost_small_serial_run(tmp_path: Path):
    runner = BenchmarkRunner(
        problem_name="synthetic_xgboost",
        optimizer_name="ours",
        n_trials=2,
        parallel_backend="serial",
        workers=1,
        seed=42,
        output_dir=tmp_path,
    )
    df, summary = runner.run()
    assert len(df) == 2
    assert summary["trials"] == 2
    assert summary["best_value"] > 0  # rmse, always positive


def test_adapter_factory():
    problem = get_problem("svm")
    adapter = create_adapter("ours", problem.search_space, problem.direction, seed=42)
    config = adapter.suggest(0)
    assert isinstance(config, dict)
    assert config


def test_suite_wiring(tmp_path: Path):
    result = run_suite(
        problems=["digits_svm"],
        optimizers=["ours"],
        backends=["serial"],
        trials=1,
        workers=1,
        seed=42,
        output_dir=tmp_path,
        build_final_report=False,
    )
    assert result["rows"] == 1
    assert result["runs"][0]["problem"] == "digits_svm"
