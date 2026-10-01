"""Friendly public interface over the AdaptiveOptimizer engine in core.py.

Two calling conventions, matching what anyone coming from Optuna/Hyperopt/Ray
Tune will already expect:

    from metatune import Study, Float, Int, Categorical

    study = Study(
        space={
            "lr": Float(1e-4, 1e-1, log=True),
            "num_layers": Int(1, 8),
            "activation": Categorical(["relu", "tanh", "gelu"]),
        },
        direction="maximize",
    )

    # (a) hand it an objective function directly
    study.optimize(lambda cfg: train_and_eval(cfg), n_trials=50)

    # (b) or drive it manually, e.g. inside your own training loop
    for _ in range(50):
        config = study.ask()
        value = train_and_eval(config)
        study.tell(value)

    print(study.best_config, study.best_value)

ask()/tell() call the same collaborators AdaptiveOptimizer.step() calls
internally (meta-controller strategy selection, the real strategy portfolio,
observers, Pareto tracking, early stopping) in the same order, just split
across two calls instead of one -- so manual and .optimize() usage produce
the same kind of search, just with the objective evaluated by you or by it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

from . import core as _core

__all__ = ["Float", "Int", "Categorical", "Study"]


@dataclass(frozen=True)
class Float:
    low: float
    high: float
    log: bool = False


@dataclass(frozen=True)
class Int:
    low: int
    high: int


@dataclass(frozen=True)
class Categorical:
    choices: Sequence[Any]


ParamSpecLike = Union[Float, Int, Categorical]


def _build_registry(space: Mapping[str, ParamSpecLike]) -> "_core.ParameterRegistry":
    registry = _core.ParameterRegistry()
    for name, spec in space.items():
        if isinstance(spec, Float):
            registry.register(_core.ContinuousParameter(name, float(spec.low), float(spec.high), log_scale=spec.log))
        elif isinstance(spec, Int):
            registry.register(_core.IntegerParameter(name, int(spec.low), int(spec.high)))
        elif isinstance(spec, Categorical):
            registry.register(_core.CategoricalParameter(name, list(spec.choices)))
        else:
            raise TypeError(
                f"unsupported search-space spec for {name!r}: {type(spec).__name__} "
                "(use metatune.Float / metatune.Int / metatune.Categorical)"
            )
    return registry


class Study:
    """A single hyperparameter search, backed by the real AdaptiveOptimizer engine
    (18 search strategies behind a meta-controller, Pareto tracking, early stopping)."""

    def __init__(self, space: Mapping[str, ParamSpecLike], direction: str = "maximize", metric_name: str = "objective"):
        if direction not in ("maximize", "minimize"):
            raise ValueError("direction must be 'maximize' or 'minimize'")
        if not space:
            raise ValueError("space must declare at least one parameter")
        self.direction = direction
        self.metric_name = metric_name
        registry = _build_registry(space)
        d = _core.ScoringDirection.MAXIMIZE if direction == "maximize" else _core.ScoringDirection.MINIMIZE
        scoring = _core.CompositeScoring([_core.ScoringMetric(metric_name, d, 1.0)])
        self._optimizer = _core.AdaptiveOptimizer(parameter_registry=registry, scoring=scoring)
        self._pending: Optional[Tuple[Any, Dict[str, Any]]] = None

    def optimize(self, objective: Callable[[Dict[str, Any]], float], n_trials: int) -> "Study":
        """Run n_trials, calling objective(config) -> float yourself is not needed here --
        the optimizer calls it for you and owns timing/scheduling internally."""
        objective_fn = _core.SingleObjective(self.metric_name, lambda values, ctx: objective(values))
        for _ in range(n_trials):
            results = self._optimizer.step(objective_fn=objective_fn, batch_size=1)
            if not results:
                break  # optimizer has converged / stopped early
        return self

    def ask(self) -> Dict[str, Any]:
        """Propose the next configuration to try. Pair with a later tell(value)."""
        opt = self._optimizer
        if opt.state in (_core.OptimizerState.CONVERGED, _core.OptimizerState.STOPPED):
            return dict(self.best_config or {})
        old_name = opt.meta_controller.active_strategy.name
        strategy = opt.meta_controller.select_best_strategy(opt.tracker)
        if strategy.name != old_name:
            for obs in opt.observers:
                obs.on_strategy_switch(old_name, strategy.name)
        proposals = strategy.propose(opt.parameter_registry, opt.tracker, opt.beliefs, 1)
        config = dict(proposals[0])
        for obs in opt.observers:
            obs.on_trial_start(f"trial_{time.time()}", config)
        self._pending = (strategy, config)
        return config

    def tell(self, value: float) -> None:
        """Report the objective value for the most recent ask()."""
        if self._pending is None:
            raise RuntimeError("tell() called without a matching ask()")
        opt = self._optimizer
        strategy, config = self._pending
        self._pending = None
        now = time.time()
        composite = opt.scoring.compute_composite_score({self.metric_name: float(value)})
        result = _core.TrialResult(
            trial_id=f"trial_{now}",
            config=config,
            objective_scores={self.metric_name: float(value)},
            composite_score=composite,
            status=_core.TrialStatus.COMPLETED,
            start_time=now,
            end_time=now,
        )
        for obs in opt.observers:
            obs.on_trial_complete(result)
        strategy.update(result, opt.beliefs)
        opt.pareto_frontier.add_solution(result)
        history = opt.tracker.scores_history
        prev_best = history[-2] if len(history) >= 2 else 0.0
        improvement = max(0.0, composite - prev_best)
        opt.meta_controller.record_strategy_performance(strategy.name, improvement)
        if len(opt.tracker.history) > 0 and len(opt.tracker.history) % 10 == 0:
            opt.self_tuner.run_self_reflection()
        should_stop, _ = opt.early_stopper.evaluate_stopping(opt.tracker, opt.beliefs)
        if should_stop:
            opt.set_state(_core.OptimizerState.CONVERGED)

    @property
    def best_config(self) -> Optional[Dict[str, Any]]:
        best = self._optimizer.get_best_configuration()
        return dict(best.config) if best else None

    @property
    def best_value(self) -> Optional[float]:
        best = self._optimizer.get_best_configuration()
        return best.composite_score if best else None

    def summary(self) -> Dict[str, Any]:
        return self._optimizer.get_summary_report()
