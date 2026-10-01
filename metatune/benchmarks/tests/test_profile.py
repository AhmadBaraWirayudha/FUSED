from pathlib import Path

from metatune.benchmarks.adapters import create_adapter
from metatune.benchmarks.problems import get_problem
from metatune.benchmarks.profiling import run_suggestion_throughput


def test_throughput_profile_writes_json(tmp_path: Path):
    problem = get_problem("svm")
    adapter = create_adapter("ours", problem.search_space, problem.direction, seed=42)
    result = run_suggestion_throughput(adapter, problem.search_space, n_calls=25, output_dir=tmp_path)
    assert result.n_calls == 25
    assert (tmp_path / "throughput.json").exists()
