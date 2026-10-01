from __future__ import annotations

from pathlib import Path
import py_compile

from app_streamlit_core import PAGES, DEFAULT_100M_DATASET, find_100m_dataset

ROOT = Path(__file__).resolve().parents[1]


def test_tb13_ui_entrypoints_compile() -> None:
    py_compile.compile(str(ROOT / 'app' / 'streamlit_app.py'), doraise=True)
    py_compile.compile(str(ROOT / 'app_streamlit_core.py'), doraise=True)


def test_tb13_navigation_is_beginner_oriented() -> None:
    assert PAGES == ('Start Here', 'Ask', 'Teach', 'Data', 'Benchmark', 'System', 'Deploy')


def test_100m_default_path_is_stable() -> None:
    assert DEFAULT_100M_DATASET.endswith('synthetic_supervised_max_100m.jsonl.gz')


def test_find_100m_dataset_prefers_canonical_path(tmp_path: Path) -> None:
    canonical = tmp_path / 'data' / 'tb10' / 'synthetic_supervised_max_100m.jsonl.gz'
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b'123')
    alternate = tmp_path / 'data' / 'tb10' / 'synthetic_max_100m.jsonl.gz'
    alternate.write_bytes(b'456')
    assert find_100m_dataset(tmp_path) == canonical.resolve()


def test_ui_has_dedicated_100m_controls() -> None:
    text = (ROOT / 'app_streamlit_core.py').read_text(encoding='utf-8')
    for marker in ('Open 100M Data', 'Audit 100M corpus', 'Ingest 100M into FUSED', 'Generate 100M-token corpus'):
        assert marker in text
