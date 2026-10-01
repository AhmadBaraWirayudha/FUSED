from __future__ import annotations

import argparse
import gzip
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator

from engine import OpenClosedLoopEngine

SCHEMA_VERSION = "1.0"
SUPPORTED_SCHEMA_VERSIONS = {"1.0", "2.0", "3.0"}


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
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as fh:
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
            if schema not in SUPPORTED_SCHEMA_VERSIONS:
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
        path = engine.memory._resolve_path(db_path)
        engine.memory.sqlite_path = path
        if engine.memory.persistent_index:
            # Keep each explicitly selected database paired with its own index
            # artifact. This prevents two supervised datasets from sharing a
            # stale manifest under the config's default index directory.
            engine.memory.persistent_index_dir = path.parent / f'{path.stem}.retrieval_index'
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


def generate_synthetic_dataset(output: str | Path, count: int = 10000, target_tokens_per_record: int = 32) -> dict[str, Any]:
    """Generate the grounded synthetic corpus with provenance and train/eval split."""
    from synthetic_data import generate_grounded_synthetic_dataset
    return generate_grounded_synthetic_dataset(output, count=count, target_tokens_per_record=target_tokens_per_record)

def generate_max_token_synthetic_dataset(output: str | Path, token_budget: int = 25_000_000, target_tokens_per_record: int = 128, seed: int = 42, train_ratio: float = 0.8) -> dict[str, Any]:
    from synthetic_data import generate_max_token_dataset
    return generate_max_token_dataset(output, token_budget, target_tokens_per_record, seed, train_ratio)

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
    synth.add_argument('--target-tokens-per-record', type=int, default=32)
    max_synth = sub.add_parser('generate-max', help='Generate a max-token streaming synthetic supervised corpus.')
    max_synth.add_argument('--output', default='data/tb10/synthetic_supervised_max.jsonl.gz')
    max_synth.add_argument('--token-budget', type=int, default=25_000_000)
    max_synth.add_argument('--target-tokens-per-record', type=int, default=128)
    max_synth.add_argument('--seed', type=int, default=42)
    max_synth.add_argument('--train-ratio', type=float, default=0.8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == 'generate':
        print(json.dumps(generate_synthetic_dataset(args.output, args.count, args.target_tokens_per_record), indent=2))
        return 0
    if args.command == 'generate-max':
        print(json.dumps(generate_max_token_synthetic_dataset(args.output, args.token_budget, args.target_tokens_per_record, args.seed, args.train_ratio), indent=2))
        return 0
    print(json.dumps(ingest_supervised_dataset(args.dataset, args.config, args.db, args.batch_size, args.max_records), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
