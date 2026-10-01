from __future__ import annotations

import json

import pytest

from evaluation import EvaluationSchemaError, evaluate_dataset, load_dataset


def test_evaluation_dataset_schema_is_valid() -> None:
    version, cases = load_dataset('data/evaluation/v1.jsonl')
    assert version == 'v1'
    assert len(cases) == 4
    assert cases[0].case_id == 'm1_exact'


def test_evaluation_schema_rejects_duplicates(tmp_path) -> None:
    path = tmp_path / 'bad.jsonl'
    record = {
        'schema_version': '1.0',
        'case_id': 'duplicate',
        'input': 'x',
        'expected': {'answer_contains': [], 'retrieved_doc_ids': []},
        'metadata': {},
    }
    path.write_text(json.dumps(record) + '\n' + json.dumps(record) + '\n', encoding='utf-8')
    with pytest.raises(EvaluationSchemaError, match='duplicate case_id'):
        load_dataset(path)


def test_evaluation_runs_and_returns_metrics(tmp_path) -> None:
    summary = evaluate_dataset('data/evaluation/v1.jsonl', 'config.yaml')
    assert summary.case_count == 4
    assert 0.0 <= summary.answer_accuracy <= 1.0
    assert 0.0 <= summary.retrieval_hit_at_1 <= 1.0
    assert 0.0 <= summary.retrieval_hit_at_k <= 1.0
    assert 0.0 <= summary.retrieval_mrr <= 1.0
    assert summary.mean_latency_ms > 0.0
    assert len(summary.cases) == 4
