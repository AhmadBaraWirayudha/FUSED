from __future__ import annotations

import json

import pytest

from fractal_router import decide_fractal_route
from gemini_backend import GeminiInteractionsBackend
from moe_model import SharedMoEBackbone


def test_fractal_confidence_selects_distinct_routes() -> None:
    high = decide_fractal_route(0.9)
    medium = decide_fractal_route(0.6)
    low = decide_fractal_route(0.2)

    assert high.route == 'direct_grounded'
    assert medium.route == 'standard'
    assert low.route == 'cautious_reasoning'
    assert high.max_new_tokens < medium.max_new_tokens < low.max_new_tokens
    assert high.thinking_level == 'low'
    assert medium.thinking_level == 'medium'
    assert low.thinking_level == 'high'


def test_fallback_output_changes_with_fractal_route() -> None:
    backend = SharedMoEBackbone.from_config({
        'model': {
            'model_name': 'fallback',
            'fallback_model_name': 'fallback',
            'retrieval_confidence_threshold': 0.5,
        },
        'knowledge_graph': {'low_confidence_threshold': 0.4},
        'quantization': {'mode': 'none'},
    })
    backend.initialize()
    structured = {
        'retrieved': [{'doc_id': 'd1', 'text': 'Ohm law: V = IR.', 'score': 0.92, 'metadata': {}}],
        'plan_actions': ['retrieve', 'finalize'],
        'subtasks': ['identify the relation'],
        'fractal_route': {'route': 'direct_grounded'},
    }
    direct = backend.generate('Task: What is Ohm law?', structured_context=structured).text
    structured['fractal_route'] = {'route': 'standard'}
    standard = backend.generate('Task: What is Ohm law?', structured_context=structured).text
    assert direct != standard
    assert 'direct-grounded route' in direct
    assert 'Closest known solved example:' in standard
    structured['fractal_route'] = {'route': 'cautious_reasoning'}
    cautious = backend.generate('Task: What is Ohm law?', structured_context=structured).text
    assert 'cautious route' in cautious


def test_gemini_applies_fractal_route(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            return json.dumps({'output_text': 'ok'}).encode('utf-8')

    def fake_urlopen(request, timeout):
        seen['payload'] = json.loads(request.data.decode('utf-8'))
        return FakeResponse()

    monkeypatch.setattr('urllib.request.urlopen', fake_urlopen)
    backend = GeminiInteractionsBackend(api_key='test-key', thinking_level='medium')
    backend.generate(
        'Prompt',
        max_new_tokens=256,
        structured_context={
            'fractal_route': {
                'route': 'cautious_reasoning',
                'thinking_level': 'high',
                'max_new_tokens': 384,
            }
        },
    )
    assert seen['payload']['generation_config']['thinking_level'] == 'high'
    assert seen['payload']['generation_config']['max_output_tokens'] == 384


def test_unified_pipeline_exposes_fractal_route(tmp_path) -> None:
    from ai_pipeline import UnifiedAIPipeline

    pipeline = UnifiedAIPipeline(config_path='config.yaml')
    db_path = tmp_path / 'engine.db'
    pipeline.closed_loop.config['paths']['sqlite_db'] = str(db_path)
    pipeline.closed_loop.memory.sqlite_path = db_path
    pipeline.initialize()
    result = pipeline.run('Solve the integral of 2x from 0 to 4.').to_dict()
    route = result['fractal']['route']
    assert route['route'] in {'direct_grounded', 'standard', 'cautious_reasoning'}
    trace = next(item for item in result['trace'] if item['name'] == 'fractal_cognition')
    assert 'confidence' in trace['data']
