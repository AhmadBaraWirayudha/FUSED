from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from ai_pipeline import UnifiedAIPipeline
from evaluation import _normalize, EvaluationSchemaError

SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class LearningCase:
    case_id: str
    input_text: str
    ideal_output: str
    probe_text: str
    expected_contains: tuple[str, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class LearningCaseResult:
    case_id: str
    baseline_correct: bool
    post_learning_exact_correct: bool
    post_learning_probe_correct: bool
    baseline_retrieved_doc_ids: tuple[str, ...]
    learned_retrieved_doc_ids: tuple[str, ...]
    lesson_doc_id: str | None
    learning_latency_ms: float


@dataclass(frozen=True)
class LearningSummary:
    schema_version: str
    dataset_version: str
    case_count: int
    baseline_accuracy: float
    post_learning_exact_accuracy: float
    post_learning_probe_accuracy: float
    exact_accuracy_delta: float
    probe_accuracy_delta_not_applicable: bool
    cases: tuple[LearningCaseResult, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["cases"] = [asdict(case) for case in self.cases]
        return payload


def load_dataset(path: str | Path) -> tuple[str, list[LearningCase]]:
    path = Path(path)
    cases: list[LearningCase] = []
    seen: set[str] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationSchemaError(f"line {line_no}: invalid JSON: {exc}") from exc
        required = {"schema_version", "case_id", "input", "ideal_output", "probe", "expected_contains", "metadata"}
        missing = required - set(raw)
        if missing:
            raise EvaluationSchemaError(f"line {line_no}: missing keys: {sorted(missing)}")
        if str(raw["schema_version"]) != SCHEMA_VERSION:
            raise EvaluationSchemaError(f"line {line_no}: unsupported schema_version")
        case_id = str(raw["case_id"]).strip()
        if not case_id or case_id in seen:
            raise EvaluationSchemaError(f"line {line_no}: invalid or duplicate case_id")
        expected = raw["expected_contains"]
        if not isinstance(expected, list) or not expected or not all(isinstance(x, str) and x for x in expected):
            raise EvaluationSchemaError(f"line {line_no}: expected_contains must contain non-empty strings")
        if not isinstance(raw["metadata"], dict):
            raise EvaluationSchemaError(f"line {line_no}: metadata must be an object")
        seen.add(case_id)
        cases.append(LearningCase(case_id, str(raw["input"]), str(raw["ideal_output"]), str(raw["probe"]), tuple(expected), dict(raw["metadata"])))
    if not cases:
        raise EvaluationSchemaError(f"dataset {path} contains no cases")
    return path.stem, cases


def _matches(output: str, expected: tuple[str, ...]) -> bool:
    haystack = _normalize(output)
    return all(_normalize(fragment) in haystack for fragment in expected)


def _new_pipeline(config_path: str | Path, db_path: Path) -> UnifiedAIPipeline:
    pipeline = UnifiedAIPipeline(config_path=config_path)
    pipeline.closed_loop.config["paths"]["sqlite_db"] = str(db_path)
    pipeline.closed_loop.memory.sqlite_path = db_path
    if pipeline.closed_loop.memory.persistent_index:
        pipeline.closed_loop.memory.persistent_index_dir = db_path.parent / f"{db_path.stem}.retrieval_index"
    pipeline.initialize()
    return pipeline


def evaluate_learning(dataset_path: str | Path = "data/evaluation/learning_v1.jsonl", config_path: str | Path = "config.yaml") -> LearningSummary:
    dataset_version, cases = load_dataset(dataset_path)
    results: list[LearningCaseResult] = []
    with TemporaryDirectory(prefix="fused-learning-") as temp_dir:
        root = Path(temp_dir)
        for index, case in enumerate(cases):
            pipeline = _new_pipeline(config_path, root / f"case-{index}.db")
            baseline = pipeline.run(case.input_text)
            baseline_correct = _matches(baseline.final_output, case.expected_contains)
            start = time.perf_counter()
            feedback = pipeline.teach_from_example(case.input_text, case.ideal_output, notes="TB06 supervised lesson")
            learning_latency_ms = (time.perf_counter() - start) * 1000.0
            exact = pipeline.run(case.input_text)
            probe = pipeline.run(case.probe_text)
            results.append(
                LearningCaseResult(
                    case_id=case.case_id,
                    baseline_correct=baseline_correct,
                    post_learning_exact_correct=_matches(exact.final_output, case.expected_contains),
                    post_learning_probe_correct=_matches(probe.final_output, case.expected_contains),
                    baseline_retrieved_doc_ids=tuple(str(d["doc_id"]) for d in baseline.closed_loop.get("retrieved", [])),
                    learned_retrieved_doc_ids=tuple(str(d["doc_id"]) for d in exact.closed_loop.get("retrieved", [])),
                    lesson_doc_id=feedback.get("fractal_training", {}).get("lesson_doc_id"),
                    learning_latency_ms=learning_latency_ms,
                )
            )
    count = len(results)
    baseline = sum(r.baseline_correct for r in results) / count
    exact = sum(r.post_learning_exact_correct for r in results) / count
    probe = sum(r.post_learning_probe_correct for r in results) / count
    return LearningSummary(
        schema_version=SCHEMA_VERSION,
        dataset_version=dataset_version,
        case_count=count,
        baseline_accuracy=baseline,
        post_learning_exact_accuracy=exact,
        post_learning_probe_accuracy=probe,
        exact_accuracy_delta=exact - baseline,
        probe_accuracy_delta_not_applicable=True,
        cases=tuple(results),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure supervised learning on FUSED after teaching corrected examples.")
    parser.add_argument("--dataset", default="data/evaluation/learning_v1.jsonl")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="docs/baselines/v0.3.1_learning.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = evaluate_learning(args.dataset, args.config)
    payload = summary.to_dict()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"dataset={summary.dataset_version}")
    print(f"cases={summary.case_count}")
    print(f"baseline_accuracy={summary.baseline_accuracy:.3f}")
    print(f"post_learning_exact_accuracy={summary.post_learning_exact_accuracy:.3f}")
    print(f"post_learning_probe_accuracy={summary.post_learning_probe_accuracy:.3f}")
    print(f"exact_accuracy_delta={summary.exact_accuracy_delta:+.3f}")
    print(f"report={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
