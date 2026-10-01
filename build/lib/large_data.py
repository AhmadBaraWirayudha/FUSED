from __future__ import annotations

import argparse
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator

from engine import OpenClosedLoopEngine

SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class SupervisedExample:
    case_id: str
    input_text: str
    ideal_output: str
    metadata: dict[str, Any]


def iter_supervised_examples(path: str | Path, max_records: int | None = None) -> Iterator[SupervisedExample]:
    path = Path(path)
    seen: set[str] = set()
    count = 0
    with path.open('r', encoding='utf-8') as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'line {line_no}: invalid JSON: {exc}') from exc
            if not isinstance(raw, dict):
                raise ValueError(f'line {line_no}: record must be an object')
            schema = str(raw.get('schema_version', SCHEMA_VERSION))
            if schema != SCHEMA_VERSION:
                raise ValueError(f'line {line_no}: unsupported schema_version={schema!r}')
            case_id = str(raw.get('case_id', '')).strip()
            input_text = str(raw.get('input', '')).strip()
            ideal = str(raw.get('ideal_output', '')).strip()
            metadata = raw.get('metadata', {})
            if not case_id or not input_text or not ideal:
                raise ValueError(f'line {line_no}: case_id, input, ideal_output are required')
            if case_id in seen:
                raise ValueError(f'line {line_no}: duplicate case_id={case_id!r}')
            if not isinstance(metadata, dict):
                raise ValueError(f'line {line_no}: metadata must be an object')
            seen.add(case_id)
            yield SupervisedExample(case_id, input_text, ideal, dict(metadata))
            count += 1
            if max_records is not None and count >= max_records:
                return


def build_engine(config_path: str | Path, db_path: str | Path | None = None) -> OpenClosedLoopEngine:
    engine = OpenClosedLoopEngine(config_path)
    if db_path is not None:
        path = Path(db_path)
        engine.memory.sqlite_path = path
    engine.initialize()
    return engine


def ingest_supervised_dataset(
    dataset_path: str | Path,
    config_path: str | Path = 'config.yaml',
    db_path: str | Path | None = None,
    batch_size: int = 500,
    max_records: int | None = None,
) -> dict[str, Any]:
    engine = build_engine(config_path, db_path)
    buffer: list[tuple[str, str, dict[str, Any]]] = []
    records = 0
    inserted = 0
    started = time.perf_counter()
    for example in iter_supervised_examples(dataset_path, max_records=max_records):
        metadata = dict(example.metadata)
        metadata.update({'case_id': example.case_id, 'dataset': str(dataset_path)})
        buffer.append((example.input_text, example.ideal_output, metadata))
        records += 1
        if len(buffer) >= batch_size:
            inserted += engine.memory.bulk_add_learning_examples(buffer, batch_size=batch_size, rebuild_index=False)
            buffer.clear()
    if buffer:
        inserted += engine.memory.bulk_add_learning_examples(buffer, batch_size=batch_size, rebuild_index=False)
    engine.memory.rebuild_index()
    elapsed = time.perf_counter() - started
    return {
        'dataset': str(dataset_path),
        'records_read': records,
        'records_inserted': inserted,
        'total_documents': engine.memory.count_documents(),
        'elapsed_s': round(elapsed, 4),
        'records_per_second': round(records / elapsed, 2) if elapsed else records,
        'db_path': str(engine.memory.sqlite_path),
    }


def generate_synthetic_dataset(output: str | Path, count: int = 10000) -> dict[str, Any]:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as fh:
        for i in range(count):
            term = f'engineering-topic-{i}'
            ideal = f'{term}: supervised reference answer {i}.'
            record = {
                'schema_version': SCHEMA_VERSION,
                'case_id': f'synth-{i:07d}',
                'input': f'Explain {term}.',
                'ideal_output': ideal,
                'metadata': {'domain': 'synthetic', 'index': i},
            }
            fh.write(json.dumps(record, ensure_ascii=False) + '\n')
    return {'output': str(output), 'records': count}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Large-data supervised ingestion utilities for FUSED.')
    sub = parser.add_subparsers(dest='command', required=True)
    ingest = sub.add_parser('ingest', help='Bulk-ingest supervised JSONL into FUSED memory.')
    ingest.add_argument('--dataset', required=True)
    ingest.add_argument('--config', default='config.yaml')
    ingest.add_argument('--db', default='data/supervised.db')
    ingest.add_argument('--batch-size', type=int, default=500)
    ingest.add_argument('--max-records', type=int, default=None)
    synth = sub.add_parser('generate', help='Generate a synthetic supervised JSONL dataset.')
    synth.add_argument('--output', default='data/synthetic_supervised_10000.jsonl')
    synth.add_argument('--count', type=int, default=10000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == 'generate':
        print(json.dumps(generate_synthetic_dataset(args.output, args.count), indent=2))
        return 0
    print(json.dumps(ingest_supervised_dataset(args.dataset, args.config, args.db, args.batch_size, args.max_records), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
