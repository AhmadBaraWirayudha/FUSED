from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class FractalRouteDecision:
    """Provider-neutral routing decision produced from FractalBrain confidence."""

    route: str
    confidence: float
    reason: str
    max_new_tokens: int
    thinking_level: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def decide_fractal_route(
    confidence: float,
    *,
    high_threshold: float = 0.75,
    low_threshold: float = 0.45,
    direct_max_new_tokens: int = 128,
    standard_max_new_tokens: int = 256,
    cautious_max_new_tokens: int = 384,
) -> FractalRouteDecision:
    """Turn FractalBrain confidence into an explicit downstream generation policy.

    High confidence chooses a concise/direct policy, medium confidence keeps the
    normal policy, and low confidence chooses a more cautious/deeper policy.
    """
    score = max(0.0, min(1.0, float(confidence)))
    if score >= high_threshold:
        return FractalRouteDecision(
            route='direct_grounded',
            confidence=score,
            reason='high FractalBrain confidence; use concise generation',
            max_new_tokens=int(direct_max_new_tokens),
            thinking_level='low',
        )
    if score < low_threshold:
        return FractalRouteDecision(
            route='cautious_reasoning',
            confidence=score,
            reason='low FractalBrain confidence; use cautious/deeper generation',
            max_new_tokens=int(cautious_max_new_tokens),
            thinking_level='high',
        )
    return FractalRouteDecision(
        route='standard',
        confidence=score,
        reason='medium FractalBrain confidence; use standard generation',
        max_new_tokens=int(standard_max_new_tokens),
        thinking_level='medium',
    )
