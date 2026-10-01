from __future__ import annotations

import json
from pathlib import Path

from memory import VectorMemoryStore
from tokenizer import TextEmbedder


def _make_store(tmp_path: Path) -> VectorMemoryStore:
    db = tmp_path / 'memory.db'
    index_dir = tmp_path / 'retrieval_index'
    return VectorMemoryStore(
        db,
        TextEmbedder('simple-hash-embedding'),
        backend='numpy',
        max_documents=100000,
        persistent_index_dir=index_dir,
        persistent_index=True,
        index_chunk_size=3,
    )


def test_persistent_index_survives_restart_and_retrieves(tmp_path: Path):
    store = _make_store(tmp_path)
    store.initialize()
    rows = [(f'doc-{i}', f'engineering lesson {i} about pumps bearings torque', {'query': f'pump {i}'}) for i in range(11)]
    assert store.bulk_add_documents(rows, batch_size=4) == 11
    assert store._persistent_backend == 'numpy'
    assert (tmp_path / 'retrieval_index' / 'manifest.json').exists()
    store.close()

    restarted = _make_store(tmp_path)
    restarted.initialize()
    assert restarted._persistent_index_valid is True
    assert restarted._persistent_backend == 'numpy'
    hits = restarted.retrieve(restarted.embedder.embed_text('engineering lesson 7 about pumps bearings torque'), top_k=3)
    assert hits
    assert hits[0].doc_id == 'doc-7'
    restarted.close()


def test_stale_index_is_rebuilt_after_write(tmp_path: Path):
    store = _make_store(tmp_path)
    store.initialize()
    store.bulk_add_documents([('doc-1', 'alpha hydraulic pump lesson', {}), ('doc-2', 'beta bearing lesson', {})], batch_size=2)
    manifest_before = json.loads((tmp_path / 'retrieval_index' / 'manifest.json').read_text(encoding='utf-8'))
    store.bulk_add_documents([('doc-3', 'gamma gearbox torque lesson', {})], batch_size=1, rebuild_index=False)
    assert store._persistent_index_valid is False
    hits = store.retrieve(store.embedder.embed_text('gamma gearbox torque lesson'), top_k=1)
    assert hits[0].doc_id == 'doc-3'
    manifest_after = json.loads((tmp_path / 'retrieval_index' / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest_after['document_count'] == 3
    assert manifest_after['max_updated_at'] >= manifest_before['max_updated_at']
    store.close()


def test_large_document_count_is_not_limited_by_active_window(tmp_path: Path):
    store = _make_store(tmp_path)
    store.max_documents = 2
    store.initialize()
    rows = [(f'doc-{i}', f'unique test token value {i}', {}) for i in range(9)]
    assert store.bulk_add_documents(rows, batch_size=3) == 9
    assert store.count_documents() == 9
    hits = store.retrieve(store.embedder.embed_text('unique test token value 8'), top_k=1)
    assert hits[0].doc_id == 'doc-8'
    store.close()

def test_cli_exposes_index_actions():
    from hybrid_cli import build_parser

    args = build_parser().parse_args(['--mode', 'index-status', '--db', 'data/supervised.db'])
    assert args.mode == 'index-status'
    assert args.db == 'data/supervised.db'


def test_index_status_does_not_bootstrap_selected_database(tmp_path: Path):
    from hybrid_cli import run_index_action

    config = tmp_path / 'config.yaml'
    source = Path(__file__).resolve().parents[1].joinpath('config.yaml').read_text(encoding='utf-8')
    config.write_text(source, encoding='utf-8')
    db = tmp_path / 'selected.db'
    payload = run_index_action(str(config), 'status', db_path=str(db))
    assert payload['document_count'] == 0
