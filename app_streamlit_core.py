from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_pipeline import UnifiedAIPipeline
from evaluation import evaluate_dataset
from hybrid_cli import run_index_action
from large_data import ingest_supervised_dataset
from learning_evaluation import evaluate_learning
from synthetic_data import (
    audit_synthetic_dataset,
    generate_grounded_synthetic_dataset,
    generate_max_token_dataset,
    validate_synthetic_quality,
)
from ui_runtime import build_runtime_config_path, gemini_key_present, package_status


APP_VERSION = 'TB13'
DEFAULT_DB = 'data/ui.db'
DEFAULT_SEMANTIC_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
DEFAULT_100M_DATASET = 'data/tb10/synthetic_supervised_max_100m.jsonl.gz'

PAGES = ('Start Here', 'Ask', 'Teach', 'Data', 'Benchmark', 'System', 'Deploy')
PAGE_HELP = {
    'Start Here': 'A simple checklist and first-run buttons.',
    'Ask': 'Ask FUSED a question.',
    'Teach': 'Give FUSED a correct answer to remember.',
    'Data': 'Prepare, audit, and load datasets.',
    'Benchmark': 'Measure FUSED instead of guessing.',
    'System': 'Inspect the retrieval index.',
    'Deploy': 'Prepare local, GitHub, or Streamlit deployment.',
}


def _json_path(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (root / p).resolve()


def _format_bytes(size: int) -> str:
    units = ('B', 'KB', 'MB', 'GB', 'TB')
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f'{value:.1f} {unit}' if unit != 'B' else f'{int(value)} B'
        value /= 1024
    return f'{size} B'


def find_100m_dataset(root: Path) -> Path | None:
    """Find an existing 100M-token corpus in the common beginner locations."""
    candidates = [
        root / DEFAULT_100M_DATASET,
        root / 'var' / 'generated' / 'synthetic_max_100m.jsonl.gz',
        root / 'data' / 'tb10' / 'synthetic_max_100m.jsonl.gz',
    ]
    for path in candidates:
        if path.exists() and path.is_file() and path.stat().st_size > 0:
            return path.resolve()
    return None


def _dataset_card(root: Path, path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            'present': False,
            'path': str(root / DEFAULT_100M_DATASET),
            'size': None,
            'manifest': None,
        }
    manifest_candidates = [
        Path(str(path) + '.manifest.json'),
        path.with_suffix('').with_suffix('.manifest.json'),
    ]
    manifest = None
    for manifest_path in manifest_candidates:
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            except Exception:
                manifest = None
            break
    return {
        'present': True,
        'path': str(path),
        'size': _format_bytes(path.stat().st_size),
        'manifest': manifest,
    }


def pipeline_for(root: Path, db_path: str, backend: str, embedding_model: str | None = None) -> UnifiedAIPipeline:
    cfg = build_runtime_config_path(root=root, db_path=db_path, backend=backend, embedding_model=embedding_model)
    pipeline = UnifiedAIPipeline(config_path=cfg)
    pipeline.initialize()
    return pipeline


def readiness(root: Path) -> dict[str, Any]:
    status = package_status(root)
    dataset = find_100m_dataset(root)
    return {
        'python': f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}',
        'python_ok': sys.version_info >= (3, 10),
        'streamlit': status['streamlit'],
        'pytest': status['pytest'],
        'semantic_embeddings': status['sentence_transformers'],
        'faiss': status['faiss'],
        'gemini_key': gemini_key_present(),
        'config_exists': (root / 'config.yaml').exists(),
        'bootstrap_exists': (root / 'data' / 'bootstrap_dataset.jsonl').exists(),
        'dataset_100m': dataset is not None,
    }


def _safe_call(fn: Callable[[], Any]) -> tuple[Any | None, str | None]:
    try:
        return fn(), None
    except Exception as exc:
        return None, f'{type(exc).__name__}: {exc}'


def _readiness_score(info: dict[str, Any]) -> tuple[int, int]:
    checks = [
        info['python_ok'],
        info['streamlit'],
        info['config_exists'] and info['bootstrap_exists'],
    ]
    return sum(bool(x) for x in checks), len(checks)


def _inject_css(st: Any) -> None:
    st.markdown(
        '''
        <style>
        .fused-hero {
            padding: 1.2rem 1.4rem;
            border: 1px solid rgba(127,127,127,.22);
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(127,127,127,.10), rgba(127,127,127,.03));
            margin-bottom: 1rem;
        }
        .fused-kicker { font-size: .78rem; text-transform: uppercase; letter-spacing: .09em; opacity: .7; }
        .fused-sub { font-size: 1.05rem; opacity: .78; margin-top: .2rem; }
        .fused-card {
            padding: 1rem;
            border: 1px solid rgba(127,127,127,.20);
            border-radius: 16px;
            min-height: 145px;
            background: rgba(127,127,127,.04);
        }
        .fused-small { font-size: .87rem; opacity: .70; }
        div[data-testid="stMetric"] { padding: .25rem 0; }
        </style>
        ''',
        unsafe_allow_html=True,
    )


def _go(st: Any, page: str) -> None:
    st.session_state['fused_page'] = page
    st.rerun()


def _show_readiness(st: Any, root: Path) -> None:
    info = readiness(root)
    ready_count, total = _readiness_score(info)
    st.caption(f'Ready status: {ready_count}/{total} core checks')
    cols = st.columns(4)
    items = [
        ('Python', info['python_ok']),
        ('Web UI', info['streamlit']),
        ('Project', info['config_exists'] and info['bootstrap_exists']),
        ('100M data', info['dataset_100m']),
    ]
    for col, (label, ok) in zip(cols, items):
        col.metric(label, 'Ready' if ok else 'Not ready')
    with st.expander('Advanced system details'):
        st.json(info)


def _render_sidebar(st: Any, root: Path) -> tuple[str, str, str | None]:
    if 'fused_page' not in st.session_state:
        st.session_state['fused_page'] = 'Start Here'

    st.sidebar.title('FUSED')
    st.sidebar.caption('AI research workspace')
    st.sidebar.radio(
        'Go to',
        PAGES,
        key='fused_page',
        format_func=lambda page: f'{page} — {PAGE_HELP[page]}',
        label_visibility='collapsed',
    )

    with st.sidebar.expander('Settings', expanded=False):
        backend_label = st.selectbox('Answer engine', ['Offline fallback', 'Gemini'], index=0)
        backend = 'gemini' if backend_label == 'Gemini' else 'fallback'
        default_db = str(root / DEFAULT_DB)
        db_path = st.text_input('Memory database', value=default_db)
        semantic_available = package_status(root)['sentence_transformers']
        embedding_options = ['Simple / offline']
        if semantic_available:
            embedding_options.append('Semantic embeddings')
        embedding_choice = st.selectbox(
            'Memory search',
            embedding_options,
            index=0,
            help='Simple search works offline. Semantic search becomes available after installing the optional package.',
        )
        if not semantic_available:
            st.caption('Semantic search is optional. Install it from the Deploy page.')
        embedding_model = DEFAULT_SEMANTIC_MODEL if embedding_choice.startswith('Semantic') else None

        if backend == 'gemini' and not gemini_key_present():
            st.warning('Gemini is selected, but GEMINI_API_KEY is missing.')

    st.sidebar.divider()
    st.sidebar.caption('Beginner path: Start Here → Ask → Teach → Benchmark → Deploy')
    return backend, db_path, embedding_model


def _render_home(
    st: Any,
    root: Path,
    backend: str,
    db_path: str,
    embedding_model: str | None,
    get_pipeline: Callable[..., UnifiedAIPipeline],
) -> None:
    st.markdown(
        '''<div class="fused-hero">
        <div class="fused-kicker">FUSED AI Research Studio</div>
        <h1 style="margin:.15rem 0 .1rem 0;">Run the project without touching Python</h1>
        <div class="fused-sub">Start small, teach it, then benchmark it.</div>
        </div>''',
        unsafe_allow_html=True,
    )

    info = readiness(root)
    _show_readiness(st, root)

    st.markdown('### Start with one of these')
    quick = st.columns(4)
    with quick[0]:
        st.markdown('<div class="fused-card"><b>Ask</b><br><span class="fused-small">Type a question and get an answer.</span></div>', unsafe_allow_html=True)
        if st.button('Ask FUSED', key='home_ask', use_container_width=True):
            _go(st, 'Ask')
    with quick[1]:
        st.markdown('<div class="fused-card"><b>Teach</b><br><span class="fused-small">Save a question + correct answer.</span></div>', unsafe_allow_html=True)
        if st.button('Teach FUSED', key='home_teach', use_container_width=True):
            _go(st, 'Teach')
    with quick[2]:
        st.markdown('<div class="fused-card"><b>100M Data</b><br><span class="fused-small">Prepare or use the large stress corpus.</span></div>', unsafe_allow_html=True)
        if st.button('Open 100M Data', key='home_100m', use_container_width=True, type='primary' if info['dataset_100m'] else 'secondary'):
            _go(st, 'Data')
    with quick[3]:
        st.markdown('<div class="fused-card"><b>Benchmark</b><br><span class="fused-small">Measure accuracy, retrieval, and latency.</span></div>', unsafe_allow_html=True)
        if st.button('Run a Benchmark', key='home_benchmark', use_container_width=True):
            _go(st, 'Benchmark')

    st.markdown('### First test')
    st.info('Recommended first action: run this before changing any settings.')
    st.code('Solve the integral of 2x from 0 to 4.', language='text')
    if st.button('Run first test', type='primary', use_container_width=True):
        with st.status('Running FUSED…', expanded=True) as status:
            started = time.perf_counter()
            result, error = _safe_call(lambda: get_pipeline(db_path, backend, embedding_model).run('Solve the integral of 2x from 0 to 4.').to_dict())
            elapsed = (time.perf_counter() - started) * 1000
            if error:
                status.update(label='Run failed', state='error')
                st.error(error)
            else:
                status.update(label=f'Run complete • {elapsed:.0f} ms', state='complete')
                st.success(result['final_output'])
                with st.expander('See what FUSED used', expanded=False):
                    route = result.get('fractal', {}).get('route', {})
                    st.json({'route': route, 'retrieved': result['closed_loop'].get('retrieved', []), 'trace': result.get('trace')})


def _render_ask(st: Any, root: Path, backend: str, db_path: str, embedding_model: str | None, get_pipeline: Callable[..., UnifiedAIPipeline]) -> None:
    st.header('Ask FUSED')
    st.caption('Type normally. You do not need special commands.')
    question = st.text_area('Question', placeholder='Example: What causes pump cavitation?', height=150, key='ask_question')
    col1, col2 = st.columns([4, 1])
    with col1:
        run = st.button('Ask FUSED', type='primary', disabled=not question.strip(), use_container_width=True)
    with col2:
        if st.button('Clear', use_container_width=True):
            st.session_state['ask_question'] = ''
            st.rerun()
    if run:
        with st.status('Thinking…', expanded=False) as status:
            result, error = _safe_call(lambda: get_pipeline(db_path, backend, embedding_model).run(question).to_dict())
            if error:
                status.update(label='Could not answer', state='error')
                st.error(error)
                return
            status.update(label='Done', state='complete')
            st.subheader('Answer')
            st.write(result['final_output'])
            c1, c2, c3 = st.columns(3)
            c1.metric('Intent', result['closed_loop']['intent'])
            c2.metric('Retrieved', len(result['closed_loop'].get('retrieved', [])))
            route = result.get('fractal', {}).get('route', {})
            c3.metric('Fractal route', route.get('route', 'unknown'))
            with st.expander('Evidence and pipeline trace'):
                st.json({'retrieved': result['closed_loop'].get('retrieved', []), 'route': route, 'reflection': result.get('reflection'), 'trace': result.get('trace')})


def _render_teach(st: Any, root: Path, backend: str, db_path: str, embedding_model: str | None, get_pipeline: Callable[..., UnifiedAIPipeline]) -> None:
    st.header('Teach FUSED')
    st.caption('Give FUSED a question and the correct answer. It stores the lesson as supervised memory.')
    q = st.text_area('Question', placeholder='What is the safe working load?', height=120, key='teach_question')
    a = st.text_area('Correct answer', placeholder='The safe working load is…', height=160, key='teach_answer')
    notes = st.text_input('Optional note', placeholder='Source, reviewer, or context', key='teach_note')
    if st.button('Save this lesson', type='primary', disabled=not q.strip() or not a.strip(), use_container_width=True):
        with st.status('Saving lesson…', expanded=False) as status:
            result, error = _safe_call(lambda: get_pipeline(db_path, backend, embedding_model).teach_from_example(q, a, notes=notes or None))
            if error:
                status.update(label='Teaching failed', state='error')
                st.error(error)
            else:
                status.update(label='Lesson saved', state='complete')
                st.success('The lesson is now in FUSED memory.')
                with st.expander('Saved lesson details'):
                    st.json(result)


def _render_100m_panel(st: Any, root: Path) -> None:
    path = find_100m_dataset(root)
    card = _dataset_card(root, path)
    st.markdown('## 100M-token data')
    st.caption('This is the large stress-test corpus. Keep it outside normal Git history.')

    if card['present']:
        manifest = card['manifest'] or {}
        cols = st.columns(3)
        cols[0].metric('Status', 'Found')
        cols[1].metric('File size', card['size'] or 'Unknown')
        cols[2].metric('Records', f"{manifest.get('records'):,}" if manifest.get('records') else 'See data card')
        st.code(card['path'], language='text')

        st.success('The 100M-token corpus is already available on this computer.')
        left, right = st.columns(2)
        with left:
            if st.button('Audit 100M corpus', key='audit_100m', use_container_width=True):
                with st.status('Auditing the 100M corpus…', expanded=False) as status:
                    report, error = _safe_call(lambda: audit_synthetic_dataset(card['path']))
                    if error:
                        status.update(label='Audit failed', state='error')
                        st.error(error)
                    else:
                        try:
                            validate_synthetic_quality(report)
                            status.update(label='Audit passed', state='complete')
                            st.success('Quality gate passed: provenance and train/eval leakage checks are clean.')
                        except Exception as exc:
                            status.update(label='Audit found issues', state='error')
                            st.error(str(exc))
                        with st.expander('Audit details'):
                            st.json(report)
        with right:
            confirm = st.checkbox('I understand that full ingestion can take significant CPU, RAM, disk, and time.', key='confirm_ingest_100m')
            if st.button('Ingest 100M into FUSED', type='primary', disabled=not confirm, key='ingest_100m', use_container_width=True):
                db = root / 'data' / 'tb10_100m.db'
                with st.status('Ingesting 100M-token corpus… This may take a long time.', expanded=True) as status:
                    result, error = _safe_call(lambda: ingest_supervised_dataset(card['path'], 'config.yaml', db, 1000, None))
                    if error:
                        status.update(label='Ingestion failed', state='error')
                        st.error(error)
                    else:
                        status.update(label='100M ingestion complete', state='complete')
                        st.success('The 100M dataset is now loaded into its own FUSED database.')
                        st.json(result)
    else:
        st.warning('The 100M-token corpus is not found in the normal locations.')
        st.write('Put the downloaded `.jsonl.gz` file here, then press Refresh:')
        st.code(str(root / DEFAULT_100M_DATASET), language='text')
        st.divider()
        st.subheader('One-click generation')
        st.write('This creates a new ~100M-token compressed corpus from the built-in synthetic generator.')
        confirmed = st.checkbox('I understand this can use significant CPU time and disk space.', key='confirm_generate_100m')
        budget = st.number_input('Token budget', min_value=1_000_000, max_value=1_000_000_000, value=100_000_000, step=1_000_000, key='100m_budget')
        if st.button('Generate 100M-token corpus', type='primary', disabled=not confirmed, use_container_width=True):
            out = root / 'var' / 'generated' / 'synthetic_max_100m.jsonl.gz'
            with st.status('Generating the 100M-token corpus… Do not close the browser tab.', expanded=True) as status:
                started = time.perf_counter()
                result, error = _safe_call(lambda: generate_max_token_dataset(out, int(budget), 512, 20261001, 0.8))
                elapsed = time.perf_counter() - started
                if error:
                    status.update(label='Generation failed', state='error')
                    st.error(error)
                else:
                    status.update(label=f'Generation complete • {elapsed:.0f} s', state='complete')
                    st.success('100M-token corpus created. Press Refresh status or rerun the app to make it the active detected corpus.')
                    st.json(result)

    if st.button('Refresh 100M data status', use_container_width=True, key='refresh_100m'):
        st.rerun()


def _render_data(st: Any, root: Path) -> None:
    st.header('Data')
    st.caption('The beginner path is: find data → audit it → ingest it → benchmark it.')
    _render_100m_panel(st, root)
    st.divider()

    with st.expander('Other data tools'):
        action = st.radio('Choose an action', ['Audit a dataset', 'Generate 10,000 cases', 'Ingest a dataset'], horizontal=False, key='other_data_action')
        if action == 'Audit a dataset':
            path_text = st.text_input('Dataset path', value=str(root / 'data' / 'tb10' / 'synthetic_supervised_max_100m.jsonl.gz'))
            if st.button('Audit dataset', use_container_width=True):
                report, error = _safe_call(lambda: audit_synthetic_dataset(path_text))
                if error:
                    st.error(error)
                else:
                    try:
                        validate_synthetic_quality(report)
                        st.success('Quality gate passed.')
                    except Exception as exc:
                        st.error(str(exc))
                    st.json(report)
        elif action == 'Generate 10,000 cases':
            out = st.text_input('Output file', value=str(root / 'var' / 'generated' / 'synthetic_10000.jsonl'))
            if st.button('Generate 10,000', type='primary', use_container_width=True):
                with st.status('Generating…', expanded=False) as status:
                    result, error = _safe_call(lambda: generate_grounded_synthetic_dataset(out, 10000, 42, 0.8, 32))
                    if error:
                        status.update(label='Generation failed', state='error')
                        st.error(error)
                    else:
                        status.update(label='Generated', state='complete')
                        st.json(result)
        else:
            dataset = st.text_input('Dataset (.jsonl or .jsonl.gz)', value=str(root / 'data' / 'tb10' / 'synthetic_supervised_max_100m.jsonl.gz'))
            db = st.text_input('Target database', value=str(root / 'data' / 'supervised.db'))
            batch = st.number_input('Batch size', min_value=10, max_value=5000, value=500, step=10)
            if st.button('Ingest dataset', type='primary', use_container_width=True):
                with st.status('Ingesting…', expanded=False) as status:
                    result, error = _safe_call(lambda: ingest_supervised_dataset(dataset, 'config.yaml', db, int(batch), None))
                    if error:
                        status.update(label='Ingestion failed', state='error')
                        st.error(error)
                    else:
                        status.update(label='Ingestion complete', state='complete')
                        st.json(result)


def _render_benchmark(st: Any, root: Path, backend: str, db_path: str, embedding_model: str | None) -> None:
    st.header('Benchmark')
    st.caption('Use this page when you want numbers, not impressions.')
    kind = st.segmented_control('Benchmark type', ['Fixed evaluation', 'Learning evaluation', 'Scale / ingestion'], default='Fixed evaluation', key='benchmark_kind')
    kind = kind or 'Fixed evaluation'

    if kind == 'Fixed evaluation':
        st.info('Small, repeatable benchmark bundled with FUSED.')
        if st.button('Run fixed benchmark', type='primary', use_container_width=True):
            with st.status('Running fixed evaluation…', expanded=False) as status:
                cfg = build_runtime_config_path(root=root, db_path=db_path, backend=backend, embedding_model=embedding_model)
                summary, error = _safe_call(lambda: evaluate_dataset(root / 'data' / 'evaluation' / 'v1.jsonl', cfg))
                if error:
                    status.update(label='Benchmark failed', state='error')
                    st.error(error)
                else:
                    status.update(label='Benchmark complete', state='complete')
                    payload = summary.to_dict()
                    cols = st.columns(4)
                    cols[0].metric('Answer accuracy', f'{payload["answer_accuracy"]:.1%}')
                    cols[1].metric('Retrieval Hit@1', f'{payload["retrieval_hit_at_1"]:.1%}')
                    cols[2].metric('MRR', f'{payload["retrieval_mrr"]:.3f}')
                    cols[3].metric('Mean latency', f'{payload["mean_latency_ms"]:.0f} ms')
                    report_dir = root / 'var' / 'benchmark'
                    report_dir.mkdir(parents=True, exist_ok=True)
                    report_path = report_dir / 'latest_fixed_evaluation.json'
                    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
                    with st.expander('Case details'):
                        st.json(payload)
                    st.download_button('Download benchmark report', data=json.dumps(payload, indent=2, ensure_ascii=False), file_name='fused_fixed_evaluation.json', mime='application/json')

    elif kind == 'Learning evaluation':
        st.info('Measures whether supervised teaching improves retrieval and answers.')
        if st.button('Run learning benchmark', type='primary', use_container_width=True):
            with st.status('Running supervised-learning evaluation…', expanded=False) as status:
                cfg = build_runtime_config_path(root=root, db_path=db_path, backend=backend, embedding_model=embedding_model)
                summary, error = _safe_call(lambda: evaluate_learning(root / 'data' / 'evaluation' / 'learning_v1.jsonl', cfg))
                if error:
                    status.update(label='Benchmark failed', state='error')
                    st.error(error)
                else:
                    status.update(label='Benchmark complete', state='complete')
                    payload = summary.to_dict()
                    cols = st.columns(3)
                    cols[0].metric('Before teaching', f'{payload["baseline_accuracy"]:.1%}')
                    cols[1].metric('After exact revisit', f'{payload["post_learning_exact_accuracy"]:.1%}')
                    cols[2].metric('After probe', f'{payload["post_learning_probe_accuracy"]:.1%}')
                    report_dir = root / 'var' / 'benchmark'
                    report_dir.mkdir(parents=True, exist_ok=True)
                    report_path = report_dir / 'latest_learning_evaluation.json'
                    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
                    with st.expander('Case details'):
                        st.json(payload)
                    st.download_button('Download learning report', data=json.dumps(payload, indent=2, ensure_ascii=False), file_name='fused_learning_evaluation.json', mime='application/json')
    else:
        dataset = st.text_input('Large dataset path', value=str(root / 'data' / 'tb10' / 'synthetic_supervised_max_100m.jsonl.gz'))
        max_records = st.number_input('Records for this scale check', min_value=100, max_value=50000, value=5000, step=500)
        st.caption('This is a bounded check. It does not ingest the entire 100M-token corpus.')
        if st.button('Run scale check', type='primary', use_container_width=True):
            with st.status('Running bounded scale check…', expanded=False) as status:
                db = root / 'var' / 'benchmarks' / 'scale.db'
                cfg = build_runtime_config_path(root=root, db_path=db, backend=backend, embedding_model=embedding_model)
                result, error = _safe_call(lambda: ingest_supervised_dataset(dataset, cfg, db, 500, int(max_records)))
                if error:
                    status.update(label='Scale check failed', state='error')
                    st.error(error)
                else:
                    status.update(label='Scale check complete', state='complete')
                    st.json(result)
                    report_dir = root / 'var' / 'benchmark'
                    report_dir.mkdir(parents=True, exist_ok=True)
                    (report_dir / 'latest_scale_check.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
                    st.download_button('Download scale report', data=json.dumps(result, indent=2, ensure_ascii=False), file_name='fused_scale_check.json', mime='application/json')


def _render_system(st: Any, root: Path, db_path: str) -> None:
    st.header('System health')
    st.caption('You normally do not need this page. Use it when FUSED says the retrieval index needs attention.')
    if st.button('Refresh status', use_container_width=True):
        st.rerun()
    status, error = _safe_call(lambda: run_index_action('config.yaml', 'status', db_path=db_path))
    if error:
        st.error(error)
    else:
        cols = st.columns(4)
        cols[0].metric('Documents', status['document_count'])
        cols[1].metric('RAM-loaded', status['active_documents_in_ram'])
        cols[2].metric('Persistent index', 'Valid' if status['persistent_index_valid'] else 'Needs rebuild')
        cols[3].metric('Backend', status['persistent_backend'] or 'none')
        with st.expander('Technical index details'):
            st.json(status)
    if st.button('Rebuild retrieval index', type='primary', use_container_width=True):
        with st.status('Rebuilding index…', expanded=False) as progress:
            result, error = _safe_call(lambda: run_index_action('config.yaml', 'rebuild', db_path=db_path))
            if error:
                progress.update(label='Rebuild failed', state='error')
                st.error(error)
            else:
                progress.update(label='Index rebuilt', state='complete')
                st.json(result)


def _render_deploy(st: Any, root: Path) -> None:
    st.header('Deploy')
    st.caption('Use this page when the app works locally and you are ready to share it.')
    checks = readiness(root)
    ready_items = [
        ('Python 3.10+', checks['python_ok']),
        ('Streamlit installed', checks['streamlit']),
        ('Source config', checks['config_exists']),
        ('Bootstrap data', checks['bootstrap_exists']),
    ]
    good = sum(ok for _, ok in ready_items)
    st.progress(good / len(ready_items), text=f'{good}/{len(ready_items)} deployment basics ready')
    for label, ok in ready_items:
        st.checkbox(label, value=ok, disabled=True)

    st.markdown('### Windows: easiest path')
    st.code('1. Double-click SETUP_FUSED.bat\n2. Double-click START_FUSED_UI.bat', language='text')
    st.success('After the browser opens, go to Benchmark and run the fixed benchmark.')

    st.markdown('### GitHub / Streamlit')
    st.write('Push the source repository. Use `app/streamlit_app.py` as the entrypoint. Keep the 100M-token dataset outside normal Git history.')
    st.code('streamlit run app/streamlit_app.py', language='text')

    st.markdown('### Gemini key')
    st.write('Never put the key in source code. Set `GEMINI_API_KEY` in the local environment or the hosting platform secret store.')
    if checks['gemini_key']:
        st.success('GEMINI_API_KEY is available to this process.')
    else:
        st.info('Gemini is optional. Offline fallback mode works without an API key.')

    st.markdown('### Optional semantic search')
    st.code('python -m pip install -e ".[semantic,faiss]" --no-build-isolation', language='text')


def render_app(st: Any) -> None:
    root = ROOT

    @st.cache_resource(show_spinner=False)
    def cached_pipeline(db_path: str, backend: str, embedding_model: str | None = None) -> UnifiedAIPipeline:
        return pipeline_for(root, db_path, backend, embedding_model)

    st.set_page_config(page_title='FUSED AI Research Studio', page_icon='F', layout='wide', initial_sidebar_state='expanded')
    _inject_css(st)
    backend, db_path, embedding_model = _render_sidebar(st, root)
    page = st.session_state.get('fused_page', 'Start Here')
    st.caption(f'FUSED {APP_VERSION} • {page}')

    if page == 'Start Here':
        _render_home(st, root, backend, db_path, embedding_model, cached_pipeline)
    elif page == 'Ask':
        _render_ask(st, root, backend, db_path, embedding_model, cached_pipeline)
    elif page == 'Teach':
        _render_teach(st, root, backend, db_path, embedding_model, cached_pipeline)
    elif page == 'Data':
        _render_data(st, root)
    elif page == 'Benchmark':
        _render_benchmark(st, root, backend, db_path, embedding_model)
    elif page == 'System':
        _render_system(st, root, db_path)
    elif page == 'Deploy':
        _render_deploy(st, root)

    st.divider()
    st.caption('Research/demo workspace. Large datasets should remain outside normal Git history.')
