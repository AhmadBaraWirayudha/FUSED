from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from language_model import GenerationOutput, LanguageModelBackend


class GeminiAPIError(RuntimeError):
    """Raised when the Gemini API cannot produce a response."""


class GeminiInteractionsBackend(LanguageModelBackend):
    """Minimal dependency-free Gemini Interactions API backend.

    Uses the public REST API directly so the base FUSED package does not need
    a vendor SDK. Authentication is read from an environment variable.
    """

    name = 'gemini'

    def __init__(
        self,
        model_name: str = 'gemini-3.8-flash',
        api_key_env: str = 'GEMINI_API_KEY',
        endpoint: str = 'https://generativelanguage.googleapis.com/v1beta/interactions',
        thinking_level: str = 'medium',
        timeout_s: float = 60.0,
        api_key: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.api_key_env = api_key_env
        self.endpoint = endpoint.rstrip('/')
        self.thinking_level = thinking_level
        self.timeout_s = float(timeout_s)
        self._api_key = api_key
        self._initialized = False

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> 'GeminiInteractionsBackend':
        model_cfg = config.get('model', {})
        gemini_cfg = model_cfg.get('gemini', {})
        return cls(
            model_name=str(model_cfg.get('model_name', 'gemini-3.8-flash')),
            api_key_env=str(gemini_cfg.get('api_key_env', 'GEMINI_API_KEY')),
            endpoint=str(
                gemini_cfg.get(
                    'endpoint',
                    'https://generativelanguage.googleapis.com/v1beta/interactions',
                )
            ),
            thinking_level=str(gemini_cfg.get('thinking_level', 'medium')).lower(),
            timeout_s=float(gemini_cfg.get('timeout_s', 60.0)),
        )

    @property
    def api_key(self) -> str | None:
        return self._api_key or os.getenv(self.api_key_env)

    def initialize(self) -> None:
        if not self.model_name:
            raise ValueError('Gemini model_name must not be empty.')
        if self.thinking_level not in {'low', 'medium', 'high'}:
            raise ValueError(
                "Gemini thinking_level must be one of 'low', 'medium', or 'high'."
            )
        if self.api_key is None:
            raise GeminiAPIError(
                f"Gemini backend requires an API key in environment variable "
                f"{self.api_key_env!r}."
            )
        self._initialized = True

    def build_prompt(
        self,
        query_text: str,
        retrieved_docs: list[str],
        intent: str,
        subtasks: list[str],
        plan_actions: list[str],
        cognitive_context: dict[str, Any] | None = None,
    ) -> str:
        ctx = '\n'.join(f'- {x}' for x in retrieved_docs[:5]) or '- none'
        tasks = '\n'.join(f'{i + 1}. {x}' for i, x in enumerate(subtasks)) or '1. solve directly'
        plan = ' -> '.join(plan_actions) or 'finalize'
        cognitive = '\n'.join(
            f'- {key}: {value}' for key, value in (cognitive_context or {}).items()
        ) or '- none'
        return f'''You are the language-generation component of the FUSED hybrid AI system.

Intent: {intent}
Open-loop plan: {plan}

Retrieved context:
{ctx}

Subtasks:
{tasks}

Cognitive context:
{cognitive}

User task:
{query_text}

Rules:
- Answer the user's task directly.
- Prefer retrieved context when it is relevant, but do not blindly copy it.
- Do not invent facts that are not supported by the context or your general knowledge.
- Show essential reasoning or calculation steps when useful.
- Keep the final answer concise and technically precise.
'''

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.95,
        repetition_penalty: float = 1.05,
        structured_context: dict[str, Any] | None = None,
    ) -> GenerationOutput:
        del temperature, top_p, repetition_penalty
        if not self._initialized:
            self.initialize()

        structured_context = structured_context or {}
        route = structured_context.get('fractal_route') or {}
        thinking_level = str(route.get('thinking_level', self.thinking_level)).lower()
        if thinking_level not in {'low', 'medium', 'high'}:
            thinking_level = self.thinking_level
        max_tokens = int(route.get('max_new_tokens', max_new_tokens))

        payload = {
            'model': self.model_name,
            'input': prompt,
            'generation_config': {
                'thinking_level': thinking_level,
                'max_output_tokens': max_tokens,
            },
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode('utf-8'),
            headers={
                'Content-Type': 'application/json',
                'x-goog-api-key': self.api_key or '',
            },
            method='POST',
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read().decode('utf-8')
        except urllib.error.HTTPError as exc:
            body = exc.read().decode('utf-8', errors='replace')
            raise GeminiAPIError(
                f'Gemini API HTTP {exc.code}: {body[:1000]}'
            ) from exc
        except urllib.error.URLError as exc:
            raise GeminiAPIError(f'Gemini API connection failed: {exc.reason}') from exc
        except TimeoutError as exc:
            raise GeminiAPIError('Gemini API request timed out.') from exc

        elapsed = time.perf_counter() - started
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise GeminiAPIError('Gemini API returned invalid JSON.') from exc

        text = self._extract_text(data).strip()
        if not text:
            raise GeminiAPIError('Gemini API returned no text output.')

        return GenerationOutput(
            text=text,
            prompt=prompt,
            used_fallback=False,
            provider=self.name,
            model=self.model_name,
            latency_s=elapsed,
        )

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        direct = data.get('output_text')
        if isinstance(direct, str) and direct.strip():
            return direct

        chunks: list[str] = []
        for output in data.get('outputs', []) or []:
            if isinstance(output, dict) and output.get('type') == 'text':
                text = output.get('text')
                if isinstance(text, str):
                    chunks.append(text)

        for step in data.get('steps', []) or []:
            if not isinstance(step, dict):
                continue
            for content in step.get('content', []) or []:
                if isinstance(content, dict) and content.get('type') == 'text':
                    text = content.get('text')
                    if isinstance(text, str):
                        chunks.append(text)

        return '\n'.join(chunks)
