from __future__ import annotations

import json
from urllib.error import HTTPError

import pytest

from gemini_backend import GeminiAPIError, GeminiInteractionsBackend


def test_gemini_build_prompt_contains_context() -> None:
    backend = GeminiInteractionsBackend(api_key='test-key')
    prompt = backend.build_prompt(
        'What is Ohm\'s law?',
        ['Ohm\'s law: V = IR.'],
        'engineering',
        ['identify the relation'],
        ['retrieve', 'finalize'],
        {'confidence': 0.8},
    )
    assert 'What is Ohm\'s law?' in prompt
    assert 'V = IR' in prompt
    assert 'confidence' in prompt


def test_gemini_requires_api_key() -> None:
    backend = GeminiInteractionsBackend(api_key_env='FUSED_TEST_MISSING_KEY')
    with pytest.raises(GeminiAPIError, match='requires an API key'):
        backend.initialize()


def test_gemini_generates_from_mocked_rest(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            return json.dumps({
                'steps': [
                    {'type': 'model_output', 'content': [{'type': 'text', 'text': '4 Ohm'}]}
                ]
            }).encode('utf-8')

    def fake_urlopen(request, timeout):
        seen['headers'] = dict(request.headers)
        seen['timeout'] = timeout
        seen['payload'] = json.loads(request.data.decode('utf-8'))
        return FakeResponse()

    monkeypatch.setattr('urllib.request.urlopen', fake_urlopen)
    backend = GeminiInteractionsBackend(
        model_name='gemini-3.8-flash',
        api_key='test-key',
        thinking_level='low',
    )
    result = backend.generate('Prompt', max_new_tokens=128)

    assert result.text == '4 Ohm'
    assert result.used_fallback is False
    assert result.provider == 'gemini'
    assert result.model == 'gemini-3.8-flash'
    assert result.latency_s is not None
    assert seen['payload']['model'] == 'gemini-3.8-flash'
    assert seen['payload']['generation_config']['thinking_level'] == 'low'
    assert seen['payload']['generation_config']['max_output_tokens'] == 128
    assert seen['headers']['X-goog-api-key'] == 'test-key'


def test_gemini_http_error_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(*_args, **_kwargs):
        raise HTTPError(
            url='https://example.test',
            code=429,
            msg='rate limit',
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr('urllib.request.urlopen', fake_urlopen)
    backend = GeminiInteractionsBackend(api_key='test-key')
    with pytest.raises(GeminiAPIError, match='HTTP 429'):
        backend.generate('Prompt')
