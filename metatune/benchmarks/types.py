"""Shared benchmark types and search-space primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Literal, Mapping, Protocol

import numpy as np


class ObjectiveProtocol(Protocol):
    def __call__(self, config: Mapping[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ParamSpec:
    kind: Literal["int", "float", "categorical"]
    low: float | None = None
    high: float | None = None
    choices: tuple[Any, ...] | None = None
    log: bool = False
    default: Any | None = None

    def sample(self, rng: np.random.Generator) -> Any:
        if self.kind == "int":
            if self.low is None or self.high is None:
                raise ValueError("int ParamSpec requires low and high")
            lo = int(self.low)
            hi = int(self.high)
            if lo > hi:
                lo, hi = hi, lo
            if self.log:
                lo_f = max(float(lo), 1e-12)
                hi_f = max(float(hi), lo_f)
                return int(round(10 ** rng.uniform(math.log10(lo_f), math.log10(hi_f))))
            return int(rng.integers(lo, hi + 1))
        if self.kind == "float":
            if self.low is None or self.high is None:
                raise ValueError("float ParamSpec requires low and high")
            lo = float(self.low)
            hi = float(self.high)
            if lo > hi:
                lo, hi = hi, lo
            if self.log:
                lo_f = max(lo, 1e-12)
                hi_f = max(hi, lo_f)
                return float(10 ** rng.uniform(math.log10(lo_f), math.log10(hi_f)))
            return float(rng.uniform(lo, hi))
        if self.kind == "categorical":
            if not self.choices:
                raise ValueError("categorical ParamSpec requires choices")
            idx = int(rng.integers(0, len(self.choices)))
            return self.choices[idx]
        raise ValueError(f"Unsupported ParamSpec kind: {self.kind}")

    def clamp(self, value: Any) -> Any:
        if self.kind == "categorical":
            if self.choices and value in self.choices:
                return value
            return self.default if self.default is not None else (self.choices[0] if self.choices else value)

        if self.kind == "int":
            lo = int(self.low if self.low is not None else value)
            hi = int(self.high if self.high is not None else value)
            if lo > hi:
                lo, hi = hi, lo
            return int(min(max(int(round(float(value))), lo), hi))

        if self.kind == "float":
            lo = float(self.low if self.low is not None else value)
            hi = float(self.high if self.high is not None else value)
            if lo > hi:
                lo, hi = hi, lo
            return float(min(max(float(value), lo), hi))

        return value


ObjectiveFn = Callable[[Mapping[str, Any]], dict[str, Any]]


@dataclass
class ProblemDefinition:
    name: str
    display_name: str
    metric_name: str
    direction: Literal["maximize", "minimize"]
    search_space: dict[str, ParamSpec]
    make_objective: Callable[[int], ObjectiveFn]
    tags: tuple[str, ...] = field(default_factory=tuple)
    description: str = ""

    def metric_is_better(self, lhs: float, rhs: float) -> bool:
        return lhs > rhs if self.direction == "maximize" else lhs < rhs

    def metric_sign(self) -> int:
        return 1 if self.direction == "maximize" else -1
