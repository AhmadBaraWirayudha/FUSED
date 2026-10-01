from pathlib import Path

import pandas as pd

from metatune.benchmarks.report import build_report


def test_report_builds_without_runs(tmp_path: Path):
    results_dir = tmp_path / "results"
    (results_dir / "runs").mkdir(parents=True)
    report = build_report(results_dir, output_path=tmp_path / "BENCHMARK_REPORT.md", figures_dir=tmp_path / "figures")
    assert "Benchmark Report" in report


def test_report_with_one_csv(tmp_path: Path):
    results_dir = tmp_path / "results"
    runs = results_dir / "runs"
    runs.mkdir(parents=True)
    df = pd.DataFrame(
        [
            {
                "trial_id": 0,
                "problem": "digits_svm",
                "optimizer": "ours",
                "backend": "serial",
                "config_json": "{}",
                "metric_name": "accuracy",
                "objective_value": 0.95,
                "total_time_sec": 0.1,
                "objective_time_sec": 0.09,
                "overhead_time_sec": 0.01,
                "suggest_time_sec": 0.001,
                "report_time_sec": 0.001,
                "worker_pid": 1,
                "timestamp_utc": "2026-07-21T00:00:00+00:00",
            }
        ]
    )
    df.to_csv(runs / "sample.csv", index=False)
    report = build_report(results_dir, output_path=tmp_path / "BENCHMARK_REPORT.md", figures_dir=tmp_path / "figures")
    assert "Runs analyzed" in report
    assert (tmp_path / "figures" / "total_time_boxplot.png").exists()
