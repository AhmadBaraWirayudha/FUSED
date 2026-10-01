from __future__ import annotations

import pytest

from language_model import LanguageModelBackend, build_language_model
from moe_model import SharedMoEBackbone


def test_fallback_backend_implements_language_model_contract() -> None:
    config = {
        'model': {
            'backend': 'fallback',
            'model_name': 'fallback',
            'fallback_model_name': 'fallback',
        },
        'quantization': {'mode': 'none'},
        'knowledge_graph': {'low_confidence_threshold': 0.4},
    }
    backend = build_language_model(config)
    assert isinstance(backend, LanguageModelBackend)
    assert isinstance(backend, SharedMoEBackbone)
    assert backend.name == 'fallback'

    backend.initialize()
    result = backend.generate('Task: test')
    assert result.text
    assert result.used_fallback is True


def test_unknown_backend_fails_explicitly() -> None:
    with pytest.raises(ValueError, match='Unsupported language-model backend'):
        build_language_model({'model': {'backend': 'not-a-real-provider'}})


def test_engine_exposes_provider_neutral_backend() -> None:
    from engine import OpenClosedLoopEngine

    engine = OpenClosedLoopEngine('config.yaml')
    assert isinstance(engine.language_model, LanguageModelBackend)
    assert engine.language_model is engine.backbone
