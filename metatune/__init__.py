"""metatune -- an adaptive hyperparameter optimizer with 18 search strategies
behind a meta-controller that switches between them based on measured
performance (TPE-style, GP-based Bayesian optimization, Hyperband, CMA-ES
approximation, population-based training, and more).

Quick start:

    from metatune import Study, Float, Int, Categorical

    study = Study(space={"lr": Float(1e-4, 1e-1, log=True)}, direction="maximize")
    study.optimize(lambda cfg: my_eval(cfg), n_trials=50)
    print(study.best_config, study.best_value)

For lower-level access (custom strategies, constraints, direct use of the
Pareto frontier, etc.), the full engine is re-exported from `metatune.core`.
"""
from __future__ import annotations

from .api import Categorical, Float, Int, Study
from . import core

__version__ = "0.1.0"

__all__ = ["Study", "Float", "Int", "Categorical", "core", "__version__"]
