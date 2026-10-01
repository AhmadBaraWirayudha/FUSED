from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerationOutput:
    """Provider-neutral result returned by a language-model backend."""

    text: str
    prompt: str
    used_fallback: bool = False
    provider: str = 'unknown'
    model: str = 'unknown'
    latency_s: float | None = None


class LanguageModelBackend(ABC):
    """Stable boundary between the FUSED pipeline and text-generation providers.

    TB03 deliberately defines the interface without loading a real model. The
    existing rule-based SharedMoEBackbone remains the default backend. Future
    adapters (local or API-hosted) can implement this contract without forcing
    changes into the retrieval/planning/fractal pipeline.
    """

    name: str = 'unknown'

    @abstractmethod
    def initialize(self) -> None:
        """Prepare the backend for generation."""

    @abstractmethod
    def build_prompt(
        self,
        query_text: str,
        retrieved_docs: list[str],
        intent: str,
        subtasks: list[str],
        plan_actions: list[str],
        cognitive_context: dict[str, Any] | None = None,
    ) -> str:
        """Build the provider-facing prompt from pipeline context."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.95,
        repetition_penalty: float = 1.05,
        structured_context: dict[str, Any] | None = None,
    ) -> GenerationOutput:
        """Generate text from a fully prepared prompt."""


def build_language_model(config: dict[str, Any]) -> LanguageModelBackend:
    """Construct the configured backend.

    TB03 supports only the existing fallback backend. Unknown backend names
    fail explicitly instead of silently selecting a different implementation.
    TB04 will add the first real provider adapter behind this same boundary.
    """
    model_cfg = config.get('model', {})
    backend = str(model_cfg.get('backend', 'fallback')).strip().lower()

    if backend in {'fallback', 'rule', 'rule_based'}:
        from moe_model import SharedMoEBackbone

        return SharedMoEBackbone.from_config(config)

    if backend in {'gemini', 'google', 'gemini_interactions'}:
        from gemini_backend import GeminiInteractionsBackend

        return GeminiInteractionsBackend.from_config(config)

    raise ValueError(
        f"Unsupported language-model backend {backend!r}. "
        "Supported backends: 'fallback' and 'gemini'."
    )
