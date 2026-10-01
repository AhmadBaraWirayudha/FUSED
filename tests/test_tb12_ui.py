from __future__ import annotations

import py_compile
from pathlib import Path

from app_streamlit_core import readiness

ROOT = Path(__file__).resolve().parents[1]


def test_ui_entrypoint_compiles() -> None:
    py_compile.compile(str(ROOT / 'app' / 'streamlit_app.py'), doraise=True)


def test_ui_core_compiles() -> None:
    py_compile.compile(str(ROOT / 'app_streamlit_core.py'), doraise=True)


def test_ui_dependency_is_pinned() -> None:
    text = (ROOT / 'app' / 'requirements.txt').read_text(encoding='utf-8')
    assert text.strip() == 'streamlit==1.64.0'


def test_ui_deployment_files_exist() -> None:
    for rel in ['SETUP_FUSED.bat', 'START_FUSED_UI.bat', 'CHECK_FUSED_READY.bat', 'Dockerfile', 'app/streamlit_app.py', 'app/requirements.txt', '.streamlit/config.toml']:
        assert (ROOT / rel).exists(), rel


def test_readiness_report_has_expected_keys() -> None:
    report = readiness(ROOT)
    for key in ['python_ok', 'streamlit', 'pytest', 'semantic_embeddings', 'faiss', 'gemini_key', 'config_exists', 'bootstrap_exists']:
        assert key in report
