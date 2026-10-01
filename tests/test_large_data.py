from __future__ import annotations

import json
import time
from pathlib import Path

from large_data import generate_synthetic_dataset, ingest_supervised_dataset, iter_supervised_examples
from synthetic_data import generate_max_token_dataset
from memory import VectorMemoryStore
from tokenizer import TextEmbedder


def _write_dataset(path: Path, n: int) -> None:
    with path.open('w', encoding='utf-8') as fh:
        for i in range(n):
            fh.write(json.dumps({
                'schema_version': '1.0',
                'case_id': f'case-{i}',
                'input': f'Explain topic {i}.',
                'ideal_output': f'Answer {i}.',
                'metadata': {'i': i},
            }) + '\n')


def test_streaming_iterator_does_not_require_full_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / 'dataset.jsonl'
    _write_dataset(dataset, 25)
    items = list(iter_supervised_examples(dataset, max_records=7))
    assert len(items) == 7
    assert items[-1].case_id == 'case-6'


def test_bulk_learning_ingestion_inserts_once_per_batch(tmp_path: Path) -> None:
    db = tmp_path / 'memory.db'
    store = VectorMemoryStore(db, TextEmbedder('simple-hash-embedding'), write_batch_size=4)
    store.initialize()
    examples = [(f'Question {i}', f'Answer {i}', {'i': i}) for i in range(11)]
    inserted = store.bulk_add_learning_examples(examples, batch_size=4, rebuild_index=False)
    store.rebuild_index()
    assert inserted == 11
    assert store.count_documents() == 11


def test_synthetic_generation_and_ingestion(tmp_path: Path) -> None:
    dataset = tmp_path / 'synthetic.jsonl'
    generate_synthetic_dataset(dataset, 50)
    started = time.perf_counter()
    report = ingest_supervised_dataset(dataset, 'config.yaml', tmp_path / 'supervised.db', batch_size=10)
    assert report['records_read'] == 50
    assert report['records_inserted'] == 50
    assert report['total_documents'] >= 54
    assert time.perf_counter() - started < 15.0


def test_gzip_schema_v3_iterator(tmp_path: Path) -> None:
    dataset = tmp_path / 'max.jsonl.gz'
    generate_max_token_dataset(dataset, token_budget=100_000, target_tokens_per_record=128, seed=42)
    items = list(iter_supervised_examples(dataset, max_records=3))
    assert len(items) == 3
    assert items[0].metadata['generator_version'].startswith('tb10-max-token-')
