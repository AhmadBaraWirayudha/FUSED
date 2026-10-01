from __future__ import annotations

import json
from pathlib import Path

from synthetic_data import audit_synthetic_dataset, generate_grounded_synthetic_dataset, generate_max_token_dataset, validate_synthetic_quality


def test_grounded_generation_has_provenance_and_holdout(tmp_path: Path) -> None:
    dataset = tmp_path / 'grounded.jsonl'
    manifest = generate_grounded_synthetic_dataset(dataset, count=1000, seed=42, train_ratio=0.8)
    assert manifest['records'] == 1000
    assert manifest['train_records'] == 800
    assert manifest['eval_records'] == 200
    report = audit_synthetic_dataset(dataset)
    validate_synthetic_quality(report)
    assert report['unique_inputs'] == 1000
    assert report['train_eval_exact_input_overlap'] == 0
    assert report['provenance_complete_ratio'] == 1.0


def test_seed_changes_parameterized_corpus(tmp_path: Path) -> None:
    first = tmp_path / 'first.jsonl'
    second = tmp_path / 'second.jsonl'
    generate_grounded_synthetic_dataset(first, count=40, seed=42)
    generate_grounded_synthetic_dataset(second, count=40, seed=43)
    first_inputs = [json.loads(line)['input'] for line in first.read_text(encoding='utf-8').splitlines()]
    second_inputs = [json.loads(line)['input'] for line in second.read_text(encoding='utf-8').splitlines()]
    assert first_inputs != second_inputs


def test_grounded_records_are_split_and_traceable(tmp_path: Path) -> None:
    dataset = tmp_path / 'grounded.jsonl'
    generate_grounded_synthetic_dataset(dataset, count=20, seed=7)
    rows = [json.loads(line) for line in dataset.read_text(encoding='utf-8').splitlines()]
    assert {row['metadata']['split'] for row in rows} == {'train', 'eval'}
    assert {row['metadata']['source_seed_id'] for row in rows} == {'m1', 'm2', 'e1', 'c1'}
    assert all(row['metadata']['synthetic'] is True for row in rows)
    assert all(row['metadata']['generation_method'] == 'deterministic_parametric_template_with_target_length' for row in rows)


def test_max_token_generation_is_gzip_streamable_and_split(tmp_path: Path) -> None:
    dataset = tmp_path / 'max.jsonl.gz'
    manifest = generate_max_token_dataset(dataset, token_budget=100_000, target_tokens_per_record=128, seed=9)
    assert manifest['estimated_total_tokens'] <= 100_000
    assert manifest['records'] > 0
    assert manifest['eval_records'] > 0
    report = audit_synthetic_dataset(dataset)
    validate_synthetic_quality(report)
    assert report['records'] == manifest['records']
    assert report['train_eval_exact_input_overlap'] == 0
