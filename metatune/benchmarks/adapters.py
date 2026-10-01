"""Optimizer adapters used by the benchmark harness."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import inspect
import threading
import time
from typing import Any, Mapping, MutableMapping

import numpy as np

from .types import ParamSpec

try:  # optional runtime dependency
    import optuna  # type: ignore
except Exception:  # pragma: no cover
    optuna = None  # type: ignore


@dataclass
class TrialState:
    trial_id: int
    config: dict[str, Any]
    value: float | None = None


class BaseAdapter:
    """Minimal adapter contract for all optimizers."""

    def __init__(self, search_space: Mapping[str, ParamSpec], direction: str, seed: int = 0) -> None:
        self.search_space = dict(search_space)
        self.direction = direction
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self._trials: dict[int, TrialState] = {}
        self._lock = threading.RLock()

    @property
    def name(self) -> str:
        return self.__class__.__name__.replace("Adapter", "").lower()

    def suggest(self, trial_id: int) -> dict[str, Any]:
        raise NotImplementedError

    def report(self, trial_id: int, value: float) -> None:
        with self._lock:
            trial = self._trials.get(trial_id)
            if trial is not None:
                trial.value = float(value)

    def final_result(self) -> dict[str, Any]:
        with self._lock:
            observed = [(tid, state.value, state.config) for tid, state in self._trials.items() if state.value is not None]
        if not observed:
            return {"best_value": None, "best_config": None, "best_trial_id": None, "trials": len(self._trials)}
        reverse = self.direction == "maximize"
        observed.sort(key=lambda item: item[1], reverse=reverse)
        best_tid, best_value, best_config = observed[0]
        return {
            "best_trial_id": best_tid,
            "best_value": best_value,
            "best_config": best_config,
            "trials": len(self._trials),
        }


class SimpleAdaptiveCore:
    """Lightweight deterministic fallback sampler.

    It is intentionally inexpensive so throughput measurements are stable even
    when the native repository optimizer or optional dependencies are missing.
    """

    def __init__(self, search_space: Mapping[str, ParamSpec], direction: str, seed: int = 0) -> None:
        self.search_space = dict(search_space)
        self.direction = direction
        self.rng = np.random.default_rng(seed)
        self.observations: list[tuple[float, dict[str, Any]]] = []

    def update(self, config: Mapping[str, Any], value: float) -> None:
        self.observations.append((float(value), dict(config)))

    def _elite(self) -> list[tuple[float, dict[str, Any]]]:
        if not self.observations:
            return []
        ordered = sorted(self.observations, key=lambda item: item[0], reverse=self.direction == "maximize")
        return ordered[: max(2, min(8, len(ordered)))]

    def suggest(self) -> dict[str, Any]:
        if len(self.observations) < 4:
            return {name: spec.sample(self.rng) for name, spec in self.search_space.items()}

        elites = self._elite()
        parent = dict(elites[int(self.rng.integers(0, len(elites)))][1])
        child: dict[str, Any] = {}
        for name, spec in self.search_space.items():
            base = parent.get(name, spec.default)
            if spec.kind == "categorical":
                if self.rng.random() < 0.78 and base in (spec.choices or ()):
                    child[name] = base
                else:
                    child[name] = spec.sample(self.rng)
                continue
            if spec.kind == "int":
                if isinstance(base, (int, float)) and self.rng.random() < 0.85:
                    lo = int(spec.low) if spec.low is not None else int(base)
                    hi = int(spec.high) if spec.high is not None else int(base)
                    if lo > hi:
                        lo, hi = hi, lo
                    span = max(1, int((hi - lo) / 6))
                    jitter = int(self.rng.integers(-span, span + 1))
                    child[name] = spec.clamp(int(base) + jitter)
                else:
                    child[name] = spec.sample(self.rng)
                continue
            if isinstance(base, (int, float)) and self.rng.random() < 0.85:
                lo = float(spec.low if spec.low is not None else base)
                hi = float(spec.high if spec.high is not None else base)
                if lo > hi:
                    lo, hi = hi, lo
                span = max((hi - lo) / 8.0, 1e-9)
                child[name] = spec.clamp(float(base) + float(self.rng.normal(0.0, span / 3.0)))
            else:
                child[name] = spec.sample(self.rng)
        return child


def _maybe_import_module(module_name: str) -> Any | None:
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def _extract_native_backend(module: Any) -> Any | None:
    """Find a callable/class in the repository module that behaves like an optimizer."""

    candidate_names = (
        "AdaptiveOptimizer",
        "Optimizer",
        "OptimizerCore",
        "SearchOptimizer",
        "HybridOptimizer",
        "AdaptiveSearch",
    )
    for name in candidate_names:
        obj = getattr(module, name, None)
        if obj is not None:
            return obj
    return None


class NativeRepoAdapter(BaseAdapter):
    """Thin wrapper around a native repository optimizer, when available."""

    def __init__(self, search_space: Mapping[str, ParamSpec], direction: str, seed: int = 0) -> None:
        super().__init__(search_space, direction, seed)
        self._native_module: Any | None = None
        self._pending_ask: tuple[Any, dict[str, Any]] | None = None
        self._native = self._load_native_backend()
        self._native_instance = self._instantiate_native()
        self._fallback_core = SimpleAdaptiveCore(search_space, direction, seed)

    def _load_native_backend(self) -> Any | None:
        module_candidates = (
            "metatune.core",
            "core",
            "adaptive_optimizer",
            "pipeline_optimizer",
            "ocle_clean_build.adaptive_optimizer",
            "ocle_clean_build.pipeline_optimizer",
        )
        for module_name in module_candidates:
            module = _maybe_import_module(module_name)
            if module is None:
                continue
            backend = _extract_native_backend(module)
            if backend is None:
                # Some repos expose a module-level factory function.
                for attr in ("create_optimizer", "build_optimizer", "make_optimizer", "suggest"):
                    candidate = getattr(module, attr, None)
                    if callable(candidate):
                        backend = candidate
                        break
            if backend is not None:
                self._native_module = module
                return backend
        return None

    def _build_native_registry_and_scoring(self, module: Any) -> tuple[Any | None, Any | None]:
        """Translate this benchmark's ParamSpec search space into the repo's own
        ParameterRegistry/CompositeScoring types (adaptive_optimizer.py exposes
        both). Without this, a class instantiated with no constructor args -- as
        the generic path below does -- silently searches its own default
        parameter set instead of this benchmark problem's search space.
        """
        registry_cls = getattr(module, "ParameterRegistry", None)
        continuous_cls = getattr(module, "ContinuousParameter", None)
        integer_cls = getattr(module, "IntegerParameter", None)
        categorical_cls = getattr(module, "CategoricalParameter", None)
        scoring_cls = getattr(module, "CompositeScoring", None)
        metric_cls = getattr(module, "ScoringMetric", None)
        direction_enum = getattr(module, "ScoringDirection", None)
        if not all([registry_cls, continuous_cls, integer_cls, categorical_cls, scoring_cls, metric_cls, direction_enum]):
            return None, None
        registry = registry_cls()
        try:
            for pname, spec in self.search_space.items():
                if spec.kind == "float":
                    registry.register(continuous_cls(pname, float(spec.low), float(spec.high), log_scale=bool(spec.log)))
                elif spec.kind == "int":
                    registry.register(integer_cls(pname, int(spec.low), int(spec.high)))
                elif spec.kind == "categorical":
                    registry.register(categorical_cls(pname, list(spec.choices or ())))
                else:
                    return None, None
        except Exception:
            return None, None
        d = direction_enum.MAXIMIZE if self.direction == "maximize" else direction_enum.MINIMIZE
        scoring = scoring_cls([metric_cls("objective", d, 1.0)])
        return registry, scoring

    def _looks_like_adaptive_optimizer(self, native: Any) -> bool:
        """Duck-types this repo's AdaptiveOptimizer shape specifically (as opposed
        to a generic suggest()/ask() optimizer), so we know it's safe to drive it
        via its real strategy/tracker/meta-controller machinery below.
        """
        return all(
            hasattr(native, attr)
            for attr in (
                "meta_controller", "parameter_registry", "tracker", "beliefs",
                "scoring", "observers", "pareto_frontier", "early_stopper", "state",
            )
        )

    def _instantiate_native(self) -> Any | None:
        if self._native is None:
            return None
        backend = self._native
        try:
            if inspect.isclass(backend):
                params = inspect.signature(backend).parameters
                if "parameter_registry" in params and self._native_module is not None:
                    registry, scoring = self._build_native_registry_and_scoring(self._native_module)
                    if registry is not None:
                        reg_kwargs: dict[str, Any] = {"parameter_registry": registry}
                        if scoring is not None and "scoring" in params:
                            reg_kwargs["scoring"] = scoring
                        try:
                            return backend(**reg_kwargs)
                        except Exception:
                            pass  # fall through to the generic path below
                kwargs: dict[str, Any] = {}
                if "search_space" in params:
                    kwargs["search_space"] = self.search_space
                if "direction" in params:
                    kwargs["direction"] = self.direction
                if "seed" in params:
                    kwargs["seed"] = self.seed
                if "rng" in params:
                    kwargs["rng"] = self.rng
                return backend(**kwargs)
            if callable(backend) and not hasattr(backend, "suggest"):
                params = inspect.signature(backend).parameters
                kwargs = {}
                if "search_space" in params:
                    kwargs["search_space"] = self.search_space
                if "direction" in params:
                    kwargs["direction"] = self.direction
                if "seed" in params:
                    kwargs["seed"] = self.seed
                return backend(**kwargs)
        except Exception:
            return None
        return backend

    def _native_suggest(self, native: Any) -> dict[str, Any] | None:
        try:
            if hasattr(native, "suggest") and callable(native.suggest):
                config = native.suggest()
                return dict(config) if isinstance(config, Mapping) else None
            if hasattr(native, "ask") and callable(native.ask):
                config = native.ask()
                return dict(config) if isinstance(config, Mapping) else None
            if self._looks_like_adaptive_optimizer(native):
                return self._strategy_level_suggest(native)
            if callable(native):
                config = native()
                return dict(config) if isinstance(config, Mapping) else None
        except Exception:
            return None
        return None

    def _strategy_level_suggest(self, native: Any) -> dict[str, Any] | None:
        """Ask-side for a real AdaptiveOptimizer. Its public API is step()/optimize(),
        which insist on owning objective evaluation themselves -- incompatible with
        this harness's ask-then-report contract. Rather than reimplementing or
        modifying that production class, this calls the exact same collaborators
        step() calls (meta_controller, strategy.propose, observers), in the same
        order, so the real strategy portfolio is exercised faithfully from outside.
        """
        module = self._native_module
        state_cls = getattr(module, "OptimizerState", None) if module is not None else None
        if state_cls is not None and native.state in (state_cls.CONVERGED, state_cls.STOPPED):
            return None
        old_strategy_name = native.meta_controller.active_strategy.name
        strategy = native.meta_controller.select_best_strategy(native.tracker)
        if strategy.name != old_strategy_name:
            for obs in native.observers:
                obs.on_strategy_switch(old_strategy_name, strategy.name)
        proposals = strategy.propose(native.parameter_registry, native.tracker, native.beliefs, 1)
        if not proposals:
            return None
        config = dict(proposals[0])
        for obs in native.observers:
            obs.on_trial_start(f"bench_{id(config)}_{time.time()}", config)
        self._pending_ask = (strategy, config)
        return config

    def _native_report(self, native: Any, trial_id: int, value: float) -> None:
        if self._pending_ask is not None and self._looks_like_adaptive_optimizer(native):
            self._strategy_level_report(native, value)
            return
        for method_name in ("report", "tell", "update", "observe"):
            method = getattr(native, method_name, None)
            if not callable(method):
                continue
            try:
                try:
                    method(trial_id=trial_id, value=float(value))
                except TypeError:
                    try:
                        method(float(value))
                    except TypeError:
                        method(trial_id, float(value))
                return
            except Exception:
                continue

    def _strategy_level_report(self, native: Any, value: float) -> None:
        """Tell-side counterpart to _strategy_level_suggest. Replicates step()'s
        post-evaluation bookkeeping (observer notification, strategy.update,
        Pareto tracking, meta-controller credit assignment, periodic self-tuning,
        early-stopping check) from outside, so the optimizer actually learns
        across trials exactly as it would under step().
        """
        module = self._native_module
        trial_result_cls = getattr(module, "TrialResult", None) if module is not None else None
        trial_status_cls = getattr(module, "TrialStatus", None) if module is not None else None
        state_cls = getattr(module, "OptimizerState", None) if module is not None else None
        if trial_result_cls is None or trial_status_cls is None or self._pending_ask is None:
            self._pending_ask = None
            return
        strategy, config = self._pending_ask
        self._pending_ask = None
        now = time.time()
        composite = native.scoring.compute_composite_score({"objective": float(value)})
        result = trial_result_cls(
            trial_id=f"bench_{now}",
            config=config,
            objective_scores={"objective": float(value)},
            composite_score=composite,
            status=trial_status_cls.COMPLETED,
            start_time=now,
            end_time=now,
        )
        for obs in native.observers:
            obs.on_trial_complete(result)
        strategy.update(result, native.beliefs)
        native.pareto_frontier.add_solution(result)
        history = native.tracker.scores_history
        prev_best = history[-2] if len(history) >= 2 else 0.0
        improvement = max(0.0, composite - prev_best)
        native.meta_controller.record_strategy_performance(strategy.name, improvement)
        if len(native.tracker.history) > 0 and len(native.tracker.history) % 10 == 0:
            native.self_tuner.run_self_reflection()
        should_stop, _ = native.early_stopper.evaluate_stopping(native.tracker, native.beliefs)
        if should_stop and state_cls is not None:
            native.set_state(state_cls.CONVERGED)

    def suggest(self, trial_id: int) -> dict[str, Any]:
        native = self._native_instance
        if native is not None:
            config = self._native_suggest(native)
            if config is not None:
                self._trials[trial_id] = TrialState(trial_id=trial_id, config=dict(config))
                return config
        config = self._fallback_core.suggest()
        self._trials[trial_id] = TrialState(trial_id=trial_id, config=dict(config))
        return config

    def report(self, trial_id: int, value: float) -> None:
        super().report(trial_id, value)
        native = self._native_instance
        if native is not None:
            self._native_report(native, trial_id, value)
        trial = self._trials.get(trial_id)
        if trial is not None:
            self._fallback_core.update(trial.config, value)


class OptunaAdapter(BaseAdapter):
    """Adapter for Optuna, with a graceful fallback."""

    def __init__(self, search_space: Mapping[str, ParamSpec], direction: str, seed: int = 0) -> None:
        super().__init__(search_space, direction, seed)
        self._trial_map: dict[int, Any] = {}
        self._study = None
        self._fallback = None
        if optuna is not None:
            sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True)
            self._study = optuna.create_study(direction=direction, sampler=sampler)
        else:
            self._fallback = SimpleAdaptiveCore(search_space, direction, seed)

    @property
    def name(self) -> str:
        return "optuna"

    def _suggest_from_trial(self, trial: Any) -> dict[str, Any]:
        config: dict[str, Any] = {}
        for name, spec in self.search_space.items():
            if spec.kind == "categorical":
                config[name] = trial.suggest_categorical(name, list(spec.choices or ()))
                continue
            if spec.kind == "int":
                low = int(spec.low if spec.low is not None else 0)
                high = int(spec.high if spec.high is not None else low)
                if low > high:
                    low, high = high, low
                if spec.log:
                    config[name] = trial.suggest_int(name, max(1, low), max(1, high), log=True)
                else:
                    config[name] = trial.suggest_int(name, low, high)
                continue
            if spec.kind == "float":
                low = float(spec.low if spec.low is not None else 0.0)
                high = float(spec.high if spec.high is not None else low)
                if low > high:
                    low, high = high, low
                if spec.log:
                    config[name] = trial.suggest_float(name, max(low, 1e-12), max(high, max(low, 1e-12)), log=True)
                else:
                    config[name] = trial.suggest_float(name, low, high)
                continue
            raise ValueError(f"Unsupported spec kind: {spec.kind}")
        return config

    def suggest(self, trial_id: int) -> dict[str, Any]:
        if self._study is not None:
            trial = self._study.ask()
            config = self._suggest_from_trial(trial)
            self._trial_map[trial_id] = trial
            self._trials[trial_id] = TrialState(trial_id=trial_id, config=dict(config))
            return config
        assert self._fallback is not None
        config = self._fallback.suggest()
        self._trials[trial_id] = TrialState(trial_id=trial_id, config=dict(config))
        return config

    def report(self, trial_id: int, value: float) -> None:
        super().report(trial_id, value)
        if self._study is not None:
            trial = self._trial_map.get(trial_id)
            if trial is not None:
                self._study.tell(trial, float(value))
        elif self._fallback is not None:
            trial = self._trials.get(trial_id)
            if trial is not None:
                self._fallback.update(trial.config, value)


class RayTuneOptunaAdapter(OptunaAdapter):
    """Optuna-backed adapter that prefers Ray Tune's OptunaSearch when available."""

    def __init__(self, search_space: Mapping[str, ParamSpec], direction: str, seed: int = 0) -> None:
        self._ray_search = None
        self._ray_metric = "score"
        self._ray_mode = "max" if direction == "maximize" else "min"
        try:
            from ray.tune.search.optuna import OptunaSearch  # type: ignore
            self._ray_search = OptunaSearch(metric=self._ray_metric, mode=self._ray_mode)
        except Exception:
            self._ray_search = None
        super().__init__(search_space, direction, seed)

    @property
    def name(self) -> str:
        return "raytune"

    def suggest(self, trial_id: int) -> dict[str, Any]:
        if self._ray_search is not None:
            try:
                config = self._ray_search.suggest(str(trial_id))
                if isinstance(config, dict):
                    self._trials[trial_id] = TrialState(trial_id=trial_id, config=dict(config))
                    self._trial_map[trial_id] = str(trial_id)
                    return config
            except Exception:
                pass
        return super().suggest(trial_id)

    def report(self, trial_id: int, value: float) -> None:
        super().report(trial_id, value)
        if self._ray_search is not None:
            try:
                self._ray_search.on_trial_complete(str(trial_id), result={self._ray_metric: float(value)})
            except Exception:
                pass


ADAPTERS = {
    "ours": NativeRepoAdapter,
    "optuna": OptunaAdapter,
    "raytune": RayTuneOptunaAdapter,
}


def create_adapter(name: str, search_space: Mapping[str, ParamSpec], direction: str, seed: int = 0) -> BaseAdapter:
    key = name.lower()
    if key not in ADAPTERS:
        raise KeyError(f"Unknown optimizer: {name}")
    return ADAPTERS[key](search_space, direction, seed)
