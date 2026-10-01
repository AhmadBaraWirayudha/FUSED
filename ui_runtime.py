from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from engine import dump_simple_yaml, parse_simple_yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def load_config(root: Path | None = None) -> dict[str, Any]:
    root = root or repo_root()
    return parse_simple_yaml((root / 'config.yaml').read_text(encoding='utf-8'))


def deep_update(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def runtime_config(
    *,
    root: Path | None = None,
    db_path: str | Path | None = None,
    backend: str | None = None,
    embedding_model: str | None = None,
) -> dict[str, Any]:
    root = root or repo_root()
    config = load_config(root)
    if db_path is not None:
        db = Path(db_path)
        if not db.is_absolute():
            db = (root / db).resolve()
        deep_update(config, {'paths': {
            'sqlite_db': str(db),
            'optimizer_db': str((db.parent / f'{db.stem}.optimizer.db').resolve()),
            'retrieval_index_dir': str((db.parent / f'{db.stem}.retrieval_index').resolve()),
            'bootstrap_dataset': str((root / config['paths']['bootstrap_dataset']).resolve()),
        }})
    if backend is not None:
        backend = backend.strip().lower()
        if backend == 'gemini':
            deep_update(config, {'model': {'backend': 'gemini'}})
        else:
            deep_update(config, {'model': {'backend': 'fallback', 'model_name': 'fallback'}})
    if embedding_model:
        deep_update(config, {'retrieval': {'embedding_model': embedding_model}})
    return config


def write_runtime_config(config: dict[str, Any], root: Path | None = None) -> Path:
    root = root or repo_root()
    out_dir = root / 'var' / 'ui'
    out_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode('utf-8')).hexdigest()[:12]
    path = out_dir / f'config-{digest}.yaml'
    if not path.exists():
        path.write_text(dump_simple_yaml(config) + '\n', encoding='utf-8')
    return path


def build_runtime_config_path(
    *,
    root: Path | None = None,
    db_path: str | Path = 'data/ui.db',
    backend: str = 'fallback',
    embedding_model: str | None = None,
) -> Path:
    config = runtime_config(root=root, db_path=db_path, backend=backend, embedding_model=embedding_model)
    return write_runtime_config(config, root=root)


def package_status(root: Path | None = None) -> dict[str, Any]:
    """Return dependency readiness without importing optional packages eagerly."""
    del root
    packages = {}
    for name in ('pytest', 'streamlit', 'sentence_transformers', 'faiss'):
        try:
            __import__(name)
            packages[name] = True
        except Exception:
            packages[name] = False
    return packages


def gemini_key_present() -> bool:
    return bool(os.getenv('GEMINI_API_KEY'))
