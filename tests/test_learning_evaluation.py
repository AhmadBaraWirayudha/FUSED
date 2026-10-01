from pathlib import Path

from learning_evaluation import evaluate_learning, load_dataset


def test_learning_dataset_is_valid() -> None:
    _, cases = load_dataset(Path("data/evaluation/learning_v1.jsonl"))
    assert len(cases) == 2
    assert all(case.ideal_output for case in cases)


def test_learning_improves_exact_revisit_accuracy(tmp_path) -> None:
    summary = evaluate_learning(
        "data/evaluation/learning_v1.jsonl",
        "config.yaml",
    )
    assert summary.baseline_accuracy == 0.0
    assert summary.post_learning_exact_accuracy == 1.0
    assert summary.exact_accuracy_delta == 1.0
    assert all(case.lesson_doc_id for case in summary.cases)
