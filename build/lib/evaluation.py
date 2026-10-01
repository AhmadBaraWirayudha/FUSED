from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable

from ai_pipeline import UnifiedAIPipeline

SCHEMA_VERSION = "1.0"


class EvaluationSchemaError(ValueError):
    """Raised when an evaluation dataset does not satisfy the versioned contract."""


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    input_text: str
    expected_answer_contains: tuple[str, ...]
    expected_doc_ids: tuple[str, ...]
    metadata: dict[str, Any]
    answer_match: str = "all"


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    input_text: str
    answer_correct: bool
    retrieval_hit_at_1: bool
    retrieval_hit_at_k: bool
    reciprocal_rank: float
    latency_ms: float
    output_preview: str
    retrieved_doc_ids: tuple[str, ...]
    retrieved_scores: tuple[float, ...]


@dataclass(frozen=True)
class EvaluationSummary:
    schema_version: str
    dataset_version: str
    config_path: str
    case_count: int
    answer_accuracy: float
    retrieval_hit_at_1: float
    retrieval_hit_at_k: float
    retrieval_mrr: float
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    max_latency_ms: float
    cases: tuple[CaseResult, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["cases"] = [asdict(case) for case in self.cases]
        return payload


def _normalize(text: str) -> str:
    return " ".join(text.casefold().strip().split())


def _validate_case(raw: dict[str, Any], line_number: int) -> EvaluationCase:
    required = {"schema_version", "case_id", "input", "expected", "metadata"}
    missing = required - set(raw)
    if missing:
        raise EvaluationSchemaError(f"line {line_number}: missing keys: {sorted(missing)}")
    if str(raw["schema_version"]) != SCHEMA_VERSION:
        raise EvaluationSchemaError(
            f"line {line_number}: unsupported schema_version={raw['schema_version']!r}; expected {SCHEMA_VERSION!r}"
        )
    if not isinstance(raw["case_id"], str) or not raw["case_id"].strip():
        raise EvaluationSchemaError(f"line {line_number}: case_id must be a non-empty string")
    if not isinstance(raw["input"], str) or not raw["input"].strip():
        raise EvaluationSchemaError(f"line {line_number}: input must be a non-empty string")
    if not isinstance(raw["expected"], dict):
        raise EvaluationSchemaError(f"line {line_number}: expected must be an object")
    if not isinstance(raw["metadata"], dict):
        raise EvaluationSchemaError(f"line {line_number}: metadata must be an object")

    expected = raw["expected"]
    answer_contains = expected.get("answer_contains", [])
    doc_ids = expected.get("retrieved_doc_ids", [])
    answer_match = expected.get("answer_match", "all")
    if not isinstance(answer_contains, list) or not all(isinstance(x, str) and x for x in answer_contains):
        raise EvaluationSchemaError(f"line {line_number}: expected.answer_contains must be a list of non-empty strings")
    if not isinstance(doc_ids, list) or not all(isinstance(x, str) and x for x in doc_ids):
        raise EvaluationSchemaError(f"line {line_number}: expected.retrieved_doc_ids must be a list of non-empty strings")
    if answer_match not in {"all", "any"}:
        raise EvaluationSchemaError(f"line {line_number}: expected.answer_match must be 'all' or 'any'")

    return EvaluationCase(
        case_id=raw["case_id"].strip(),
        input_text=raw["input"],
        expected_answer_contains=tuple(answer_contains),
        expected_doc_ids=tuple(doc_ids),
        metadata=dict(raw["metadata"]),
        answer_match=answer_match,
    )


def load_dataset(path: str | Path) -> tuple[str, list[EvaluationCase]]:
    path = Path(path)
    dataset_version = path.stem
    cases: list[EvaluationCase] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            raw = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise EvaluationSchemaError(f"line {line_number}: invalid JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise EvaluationSchemaError(f"line {line_number}: each record must be a JSON object")
        case = _validate_case(raw, line_number)
        if case.case_id in seen_ids:
            raise EvaluationSchemaError(f"line {line_number}: duplicate case_id={case.case_id!r}")
        seen_ids.add(case.case_id)
        cases.append(case)
    if not cases:
        raise EvaluationSchemaError(f"dataset {path} contains no evaluation cases")
    return dataset_version, cases


def _answer_matches(case: EvaluationCase, output: str) -> bool:
    if not case.expected_answer_contains:
        return True
    haystack = _normalize(output)
    checks = [_normalize(fragment) in haystack for fragment in case.expected_answer_contains]
    return all(checks) if case.answer_match == "all" else any(checks)


def _reciprocal_rank(expected_ids: Iterable[str], retrieved_ids: Iterable[str]) -> float:
    expected = set(expected_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in expected:
            return 1.0 / rank
    return 0.0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return float(ordered[position])


def _new_pipeline(config_path: str | Path, db_path: Path) -> UnifiedAIPipeline:
    pipeline = UnifiedAIPipeline(config_path=config_path)
    pipeline.closed_loop.config["paths"]["sqlite_db"] = str(db_path)
    pipeline.closed_loop.memory.sqlite_path = db_path
    pipeline.initialize()
    return pipeline


def evaluate_dataset(
    dataset_path: str | Path,
    config_path: str | Path = "config.yaml",
) -> EvaluationSummary:
    dataset_version, cases = load_dataset(dataset_path)
    results: list[CaseResult] = []

    # Every case runs in a fresh local database and a fresh pipeline so that
    # case ordering, prior feedback, and session state cannot contaminate the
    # benchmark. Timing starts after initialization and measures only one
    # warm pipeline request.
    with TemporaryDirectory(prefix="fused-eval-") as temp_dir:
        temp_root = Path(temp_dir)
        for index, case in enumerate(cases):
            pipeline = _new_pipeline(config_path, temp_root / f"case-{index}.db")
            start = time.perf_counter()
            result = pipeline.run(case.input_text)
            latency_ms = (time.perf_counter() - start) * 1000.0

            retrieved = result.closed_loop.get("retrieved", [])
            retrieved_ids = tuple(str(item.get("doc_id")) for item in retrieved)
            retrieved_scores = tuple(float(item.get("score", 0.0)) for item in retrieved)
            expected = set(case.expected_doc_ids)
            hit_at_1 = bool(retrieved_ids and expected and retrieved_ids[0] in expected)
            hit_at_k = bool(expected.intersection(retrieved_ids))

            results.append(
                CaseResult(
                    case_id=case.case_id,
                    input_text=case.input_text,
                    answer_correct=_answer_matches(case, result.final_output),
                    retrieval_hit_at_1=hit_at_1,
                    retrieval_hit_at_k=hit_at_k,
                    reciprocal_rank=_reciprocal_rank(case.expected_doc_ids, retrieved_ids),
                    latency_ms=latency_ms,
                    output_preview=re.sub(r"\s+", " ", result.final_output).strip()[:240],
                    retrieved_doc_ids=retrieved_ids,
                    retrieved_scores=retrieved_scores,
                )
            )

    latencies = [item.latency_ms for item in results]
    count = len(results)
    return EvaluationSummary(
        schema_version=SCHEMA_VERSION,
        dataset_version=dataset_version,
        config_path=str(config_path),
        case_count=count,
        answer_accuracy=sum(item.answer_correct for item in results) / count,
        retrieval_hit_at_1=sum(item.retrieval_hit_at_1 for item in results) / count,
        retrieval_hit_at_k=sum(item.retrieval_hit_at_k for item in results) / count,
        retrieval_mrr=sum(item.reciprocal_rank for item in results) / count,
        mean_latency_ms=statistics.fmean(latencies),
        median_latency_ms=statistics.median(latencies),
        p95_latency_ms=_percentile(latencies, 0.95),
        max_latency_ms=max(latencies),
        cases=tuple(results),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the FUSED pipeline against a versioned JSONL dataset.")
    parser.add_argument("--dataset", default="data/evaluation/v1.jsonl")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="docs/baselines/v0.3.1_evaluation.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = evaluate_dataset(args.dataset, args.config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"dataset={summary.dataset_version}")
    print(f"cases={summary.case_count}")
    print(f"answer_accuracy={summary.answer_accuracy:.3f}")
    print(f"retrieval_hit_at_1={summary.retrieval_hit_at_1:.3f}")
    print(f"retrieval_hit_at_k={summary.retrieval_hit_at_k:.3f}")
    print(f"retrieval_mrr={summary.retrieval_mrr:.3f}")
    print(f"mean_latency_ms={summary.mean_latency_ms:.2f}")
    print(f"median_latency_ms={summary.median_latency_ms:.2f}")
    print(f"p95_latency_ms={summary.p95_latency_ms:.2f}")
    print(f"max_latency_ms={summary.max_latency_ms:.2f}")
    print(f"report={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
