"""Offline-safe CPU-only benchmark problems."""

from __future__ import annotations

from functools import lru_cache, partial
from typing import Any

import numpy as np
from sklearn.datasets import fetch_openml, load_breast_cancer, load_diabetes, load_digits, make_classification
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .types import ParamSpec, ProblemDefinition

try:  # optional
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None  # type: ignore

try:  # optional
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None  # type: ignore

try:  # optional
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except Exception:  # pragma: no cover
    SMOTE = None  # type: ignore
    ImbPipeline = None  # type: ignore


@lru_cache(maxsize=8)
def _digits_arrays() -> tuple[np.ndarray, np.ndarray]:
    data = load_digits()
    return data.data.astype(np.float32), data.target.astype(np.int64)


@lru_cache(maxsize=8)
def _breast_cancer_arrays() -> tuple[np.ndarray, np.ndarray]:
    data = load_breast_cancer()
    return data.data.astype(np.float32), data.target.astype(np.int64)


@lru_cache(maxsize=8)
def _diabetes_arrays() -> tuple[np.ndarray, np.ndarray]:
    data = load_diabetes()
    return data.data.astype(np.float32), data.target.astype(np.float32)


@lru_cache(maxsize=8)
def _synthetic_regression_arrays(
    seed: int, n_samples: int = 4000, n_features: int = 24, n_informative: int = 12, noise: float = 15.0
) -> tuple[np.ndarray, np.ndarray]:
    """Fully synthetic regression data (make_regression), unlike _diabetes_arrays' fixed
    442-sample real dataset -- lets dataset size/noise/dimensionality be dialed up or down
    directly, which is the point: isolating whether a timing or search-quality effect is about
    the optimizer/model or just an artifact of one small, fixed, real dataset.
    """
    from sklearn.datasets import make_regression

    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=seed,
    )
    return X.astype(np.float32), y.astype(np.float32)


@lru_cache(maxsize=8)
def _imbalanced_arrays(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=4000,
        n_features=24,
        n_informative=10,
        n_redundant=4,
        n_clusters_per_class=2,
        weights=[0.92, 0.08],
        flip_y=0.015,
        class_sep=1.0,
        random_state=seed,
    )
    return X.astype(np.float32), y.astype(np.int64)


@lru_cache(maxsize=16)
def _mnist_like_subset(seed: int = 42, n_samples: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    """Prefer real OpenML MNIST, then fall back to a deterministic offline surrogate."""

    try:
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X = np.asarray(mnist.data, dtype=np.float32) / 255.0
        y = np.asarray(mnist.target, dtype=np.int64)
        rng = np.random.default_rng(seed)
        if len(X) > n_samples:
            idx = rng.choice(len(X), size=n_samples, replace=False)
            X = X[idx]
            y = y[idx]
        return X, y
    except Exception:
        digits_X, digits_y = _digits_arrays()
        reps = int(np.ceil(n_samples / len(digits_X)))
        X = np.repeat(digits_X, reps, axis=0)[:n_samples].copy()
        y = np.repeat(digits_y, reps, axis=0)[:n_samples].copy()
        rng = np.random.default_rng(seed)
        scale = rng.uniform(0.85, 1.15, size=(len(X), 1)).astype(np.float32)
        noise = rng.normal(0.0, 0.6, size=X.shape).astype(np.float32)
        X = X * scale + noise
        return X.astype(np.float32), y.astype(np.int64)


def _split_arrays(X: np.ndarray, y: np.ndarray, *, seed: int, test_size: float = 0.25):
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def _make_svm_digits(seed: int):
    X, y = _digits_arrays()
    X_train, X_valid, y_train, y_valid = _split_arrays(X, y, seed=seed, test_size=0.25)

    def objective(config: dict[str, Any]) -> dict[str, Any]:
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svc",
                    SVC(
                        C=float(config.get("C", 1.0)),
                        gamma=str(config.get("gamma", "scale")),
                        kernel=str(config.get("kernel", "rbf")),
                        degree=int(config.get("degree", 3)),
                        coef0=float(config.get("coef0", 0.0)),
                        random_state=seed,
                    ),
                ),
            ]
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_valid)
        acc = accuracy_score(y_valid, pred)
        return {"accuracy": float(acc), "loss": float(1.0 - acc)}

    return objective


def _make_breast_cancer_lgbm(seed: int):
    X, y = _breast_cancer_arrays()
    X_train, X_valid, y_train, y_valid = _split_arrays(X, y, seed=seed, test_size=0.25)

    def objective(config: dict[str, Any]) -> dict[str, Any]:
        max_depth = int(config.get("max_depth", -1))
        if LGBMClassifier is not None:
            model = LGBMClassifier(
                n_estimators=int(config.get("n_estimators", 200)),
                learning_rate=float(config.get("learning_rate", 0.05)),
                num_leaves=int(config.get("num_leaves", 31)),
                max_depth=max_depth,
                subsample=float(config.get("subsample", 0.9)),
                colsample_bytree=float(config.get("colsample_bytree", 0.9)),
                reg_alpha=float(config.get("reg_alpha", 0.0)),
                reg_lambda=float(config.get("reg_lambda", 0.0)),
                random_state=seed,
                n_jobs=1,
                verbosity=-1,
            )
        else:
            model = HistGradientBoostingClassifier(
                learning_rate=float(config.get("learning_rate", 0.05)),
                max_depth=None if max_depth < 0 else max_depth,
                max_leaf_nodes=int(config.get("num_leaves", 31)),
                l2_regularization=float(config.get("reg_lambda", 0.0)),
                random_state=seed,
            )
        model.fit(X_train, y_train)
        pred = model.predict(X_valid)
        acc = accuracy_score(y_valid, pred)
        return {"accuracy": float(acc), "loss": float(1.0 - acc)}

    return objective


def _make_diabetes_xgboost(seed: int):
    X, y = _diabetes_arrays()
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=seed)
    return _xgb_rmse_objective(X_train, y_train, X_valid, y_valid, seed)


def _build_xgb_or_fallback_model(config: dict[str, Any], seed: int):
    max_depth = int(config.get("max_depth", 4))
    if XGBRegressor is not None:
        return XGBRegressor(
            n_estimators=int(config.get("n_estimators", 300)),
            learning_rate=float(config.get("learning_rate", 0.05)),
            max_depth=max_depth,
            subsample=float(config.get("subsample", 0.9)),
            colsample_bytree=float(config.get("colsample_bytree", 0.9)),
            min_child_weight=float(config.get("min_child_weight", 1.0)),
            reg_alpha=float(config.get("reg_alpha", 0.0)),
            reg_lambda=float(config.get("reg_lambda", 1.0)),
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
        )
    return HistGradientBoostingRegressor(
        learning_rate=float(config.get("learning_rate", 0.05)),
        max_depth=max_depth,
        max_leaf_nodes=int(config.get("num_leaves", 31)),
        l2_regularization=float(config.get("reg_lambda", 1.0)),
        random_state=seed,
    )


def _xgb_rmse_objective(X_train: np.ndarray, y_train: np.ndarray, X_valid: np.ndarray, y_valid: np.ndarray, seed: int):
    def objective(config: dict[str, Any]) -> dict[str, Any]:
        model = _build_xgb_or_fallback_model(config, seed)
        model.fit(X_train, y_train)
        pred = model.predict(X_valid)
        rmse = float(np.sqrt(mean_squared_error(y_valid, pred)))
        return {"rmse": rmse, "loss": rmse}

    return objective


def _make_synthetic_xgboost(seed: int, n_samples: int = 4000, n_features: int = 24):
    X, y = _synthetic_regression_arrays(seed, n_samples=n_samples, n_features=n_features)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=seed)
    return _xgb_rmse_objective(X_train, y_train, X_valid, y_valid, seed)


def _make_realistic_synthetic_xgboost(seed: int, method: str, n_samples: int = 1600):
    """Same XGBoost model/search space as synthetic_xgboost, but the synthetic
    data is generated to resemble the *real* diabetes dataset's joint
    distribution (see synthetic_data.py), rather than make_regression's
    arbitrary linear-plus-noise structure.
    """
    from metatune.benchmarks.synthetic_data import generate_like

    real_X, real_y = _diabetes_arrays()
    synth_X, synth_y = generate_like(method, real_X, real_y, n_samples=n_samples, seed=seed)
    X_train, X_valid, y_train, y_valid = train_test_split(synth_X, synth_y, test_size=0.2, random_state=seed)
    return _xgb_rmse_objective(X_train, y_train, X_valid, y_valid, seed)


def _make_mnist_like_mlp(seed: int):
    X, y = _mnist_like_subset(seed=seed, n_samples=5000)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    def objective(config: dict[str, Any]) -> dict[str, Any]:
        hidden_units = int(config.get("hidden_units", 128))
        n_layers = int(config.get("n_layers", 2))
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=tuple([hidden_units] * max(1, n_layers)),
                        activation=str(config.get("activation", "relu")),
                        alpha=float(config.get("alpha", 1e-4)),
                        learning_rate_init=float(config.get("learning_rate_init", 1e-3)),
                        batch_size=int(config.get("batch_size", 64)),
                        max_iter=int(config.get("max_iter", 60)),
                        random_state=seed,
                        solver="adam",
                        early_stopping=False,
                        n_iter_no_change=10,
                    ),
                ),
            ]
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_valid)
        acc = accuracy_score(y_valid, pred)
        return {"accuracy": float(acc), "loss": float(1.0 - acc)}

    return objective


def _make_imbalanced_pipeline(seed: int):
    X, y = _imbalanced_arrays(seed=seed)
    X_train, X_valid, y_train, y_valid = _split_arrays(X, y, seed=seed, test_size=0.25)

    def objective(config: dict[str, Any]) -> dict[str, Any]:
        base = LogisticRegression(
            C=float(config.get("C", 1.0)),
            max_iter=500,
            solver="lbfgs",
            n_jobs=1,
            random_state=seed,
        )
        if ImbPipeline is not None and SMOTE is not None:
            sampling_strategy = float(config.get("sampling_strategy", 0.5))
            sampling_strategy = float(min(max(sampling_strategy, 0.05), 0.95))
            model = ImbPipeline(
                [
                    (
                        "smote",
                        SMOTE(
                            sampling_strategy=sampling_strategy,
                            random_state=seed,
                        ),
                    ),
                    ("clf", base),
                ]
            )
        else:
            model = Pipeline([("scaler", StandardScaler()), ("clf", base)])
        model.fit(X_train, y_train)
        pred = model.predict(X_valid)
        score = f1_score(y_valid, pred)
        return {"f1": float(score), "loss": float(1.0 - score)}

    return objective


def _build_problem_specs() -> dict[str, ProblemDefinition]:
    return {
        "digits_svm": ProblemDefinition(
            name="digits_svm",
            display_name="Digits SVM",
            metric_name="accuracy",
            direction="maximize",
            search_space={
                "C": ParamSpec("float", low=1e-3, high=50.0, log=True, default=1.0),
                "gamma": ParamSpec("categorical", choices=("scale", "auto"), default="scale"),
                "kernel": ParamSpec("categorical", choices=("rbf", "poly"), default="rbf"),
                "degree": ParamSpec("int", low=2, high=5, default=3),
                "coef0": ParamSpec("float", low=0.0, high=2.0, default=0.0),
            },
            make_objective=_make_svm_digits,
            tags=("classification", "offline", "small"),
            description="Digits classification with an SVM pipeline.",
        ),
        "breast_cancer_lightgbm": ProblemDefinition(
            name="breast_cancer_lightgbm",
            display_name="Breast Cancer LightGBM",
            metric_name="accuracy",
            direction="maximize",
            search_space={
                "n_estimators": ParamSpec("int", low=50, high=350, default=200),
                "learning_rate": ParamSpec("float", low=0.01, high=0.2, log=True, default=0.05),
                "num_leaves": ParamSpec("int", low=16, high=63, default=31),
                "max_depth": ParamSpec("int", low=-1, high=10, default=-1),
                "subsample": ParamSpec("float", low=0.7, high=1.0, default=0.9),
                "colsample_bytree": ParamSpec("float", low=0.7, high=1.0, default=0.9),
                "reg_alpha": ParamSpec("float", low=0.0, high=3.0, default=0.0),
                "reg_lambda": ParamSpec("float", low=0.0, high=5.0, default=0.0),
            },
            make_objective=_make_breast_cancer_lgbm,
            tags=("classification", "boosting", "offline"),
            description="Breast cancer classification with LightGBM or a HistGradientBoosting fallback.",
        ),
        "diabetes_xgboost": ProblemDefinition(
            name="diabetes_xgboost",
            display_name="Diabetes XGBoost",
            metric_name="rmse",
            direction="minimize",
            search_space={
                "n_estimators": ParamSpec("int", low=50, high=500, default=300),
                "learning_rate": ParamSpec("float", low=0.005, high=0.3, log=True, default=0.05),
                "max_depth": ParamSpec("int", low=2, high=8, default=4),
                "subsample": ParamSpec("float", low=0.7, high=1.0, default=0.9),
                "colsample_bytree": ParamSpec("float", low=0.7, high=1.0, default=0.9),
                "min_child_weight": ParamSpec("float", low=0.5, high=8.0, log=True, default=1.0),
                "reg_alpha": ParamSpec("float", low=0.0, high=3.0, default=0.0),
                "reg_lambda": ParamSpec("float", low=0.1, high=8.0, log=True, default=1.0),
            },
            make_objective=_make_diabetes_xgboost,
            tags=("regression", "boosting", "offline"),
            description="Diabetes regression with XGBoost or a HistGradientBoosting fallback.",
        ),
        "synthetic_xgboost": ProblemDefinition(
            name="synthetic_xgboost",
            display_name="Synthetic XGBoost",
            metric_name="rmse",
            direction="minimize",
            search_space={
                "n_estimators": ParamSpec("int", low=50, high=500, default=300),
                "learning_rate": ParamSpec("float", low=0.005, high=0.3, log=True, default=0.05),
                "max_depth": ParamSpec("int", low=2, high=8, default=4),
                "subsample": ParamSpec("float", low=0.7, high=1.0, default=0.9),
                "colsample_bytree": ParamSpec("float", low=0.7, high=1.0, default=0.9),
                "min_child_weight": ParamSpec("float", low=0.5, high=8.0, log=True, default=1.0),
                "reg_alpha": ParamSpec("float", low=0.0, high=3.0, default=0.0),
                "reg_lambda": ParamSpec("float", low=0.1, high=8.0, log=True, default=1.0),
            },
            make_objective=_make_synthetic_xgboost,
            tags=("regression", "boosting", "synthetic"),
            description=(
                "make_regression-generated synthetic data (4000 samples, 24 features, "
                "12 informative) with XGBoost or a HistGradientBoosting fallback -- same "
                "model and search space as diabetes_xgboost, but a controllable, larger "
                "dataset instead of the fixed 442-sample real one."
            ),
        ),
        **{
            f"synthetic_xgboost_{method}": ProblemDefinition(
                name=f"synthetic_xgboost_{method}",
                display_name=f"Synthetic XGBoost ({method}-like)",
                metric_name="rmse",
                direction="minimize",
                search_space={
                    "n_estimators": ParamSpec("int", low=50, high=500, default=300),
                    "learning_rate": ParamSpec("float", low=0.005, high=0.3, log=True, default=0.05),
                    "max_depth": ParamSpec("int", low=2, high=8, default=4),
                    "subsample": ParamSpec("float", low=0.7, high=1.0, default=0.9),
                    "colsample_bytree": ParamSpec("float", low=0.7, high=1.0, default=0.9),
                    "min_child_weight": ParamSpec("float", low=0.5, high=8.0, log=True, default=1.0),
                    "reg_alpha": ParamSpec("float", low=0.0, high=3.0, default=0.0),
                    "reg_lambda": ParamSpec("float", low=0.1, high=8.0, log=True, default=1.0),
                },
                make_objective=partial(_make_realistic_synthetic_xgboost, method=method),
                tags=("regression", "boosting", "synthetic", "real-data-like"),
                description=(
                    f"Synthetic data generated to resemble the real diabetes dataset's joint "
                    f"distribution via {method} (see synthetic_data.py), rather than "
                    f"make_regression's arbitrary structure. Same XGBoost model/search space "
                    f"as diabetes_xgboost and synthetic_xgboost."
                ),
            )
            for method in ("gaussian", "gmm", "kde", "bootstrap")
        },
        "mnist_like_mlp": ProblemDefinition(
            name="mnist_like_mlp",
            display_name="MNIST 5k MLP",
            metric_name="accuracy",
            direction="maximize",
            search_space={
                "hidden_units": ParamSpec("int", low=64, high=256, default=128),
                "n_layers": ParamSpec("int", low=1, high=3, default=2),
                "activation": ParamSpec("categorical", choices=("relu", "tanh"), default="relu"),
                "alpha": ParamSpec("float", low=1e-6, high=1e-2, log=True, default=1e-4),
                "learning_rate_init": ParamSpec("float", low=1e-4, high=5e-2, log=True, default=1e-3),
                "batch_size": ParamSpec("categorical", choices=(32, 64, 128), default=64),
                "max_iter": ParamSpec("int", low=30, high=90, default=60),
            },
            make_objective=_make_mnist_like_mlp,
            tags=("classification", "neural-net", "offline"),
            description="MNIST 5k subset or deterministic digits-derived fallback.",
        ),
        "imbalanced_pipeline": ProblemDefinition(
            name="imbalanced_pipeline",
            display_name="Imbalanced Pipeline",
            metric_name="f1",
            direction="maximize",
            search_space={
                "C": ParamSpec("float", low=1e-3, high=20.0, log=True, default=1.0),
                "sampling_strategy": ParamSpec("float", low=0.2, high=0.8, default=0.5),
            },
            make_objective=_make_imbalanced_pipeline,
            tags=("classification", "imbalance", "pipeline"),
            description="Imbalanced classification with SMOTE if available, otherwise a local fallback.",
        ),
    }


_PROBLEM_SPECS = _build_problem_specs()
_ALIASES = {
    "svm": "digits_svm",
    "digits_svm": "digits_svm",
    "breast_cancer": "breast_cancer_lightgbm",
    "lightgbm": "breast_cancer_lightgbm",
    "breast_cancer_lightgbm": "breast_cancer_lightgbm",
    "diabetes": "diabetes_xgboost",
    "xgboost": "diabetes_xgboost",
    "diabetes_xgboost": "diabetes_xgboost",
    "synthetic": "synthetic_xgboost",
    "synth": "synthetic_xgboost",
    "synth_xgboost": "synthetic_xgboost",
    "synthetic_xgboost": "synthetic_xgboost",
    "synthetic_gaussian": "synthetic_xgboost_gaussian",
    "synth_gaussian": "synthetic_xgboost_gaussian",
    "synthetic_xgboost_gaussian": "synthetic_xgboost_gaussian",
    "synthetic_gmm": "synthetic_xgboost_gmm",
    "synth_gmm": "synthetic_xgboost_gmm",
    "synthetic_xgboost_gmm": "synthetic_xgboost_gmm",
    "synthetic_kde": "synthetic_xgboost_kde",
    "synth_kde": "synthetic_xgboost_kde",
    "synthetic_xgboost_kde": "synthetic_xgboost_kde",
    "synthetic_bootstrap": "synthetic_xgboost_bootstrap",
    "synth_bootstrap": "synthetic_xgboost_bootstrap",
    "synthetic_xgboost_bootstrap": "synthetic_xgboost_bootstrap",
    "mnist": "mnist_like_mlp",
    "mnist_like_mlp": "mnist_like_mlp",
    "imbalanced": "imbalanced_pipeline",
    "imbalanced_pipeline": "imbalanced_pipeline",
}


def _canonical(name: str) -> str:
    key = name.lower()
    canonical = _ALIASES.get(key)
    if canonical is None:
        raise KeyError(f"Unknown problem: {name}")
    return canonical


def get_problem(name: str) -> ProblemDefinition:
    return _PROBLEM_SPECS[_canonical(name)]


def get_problem_names() -> list[str]:
    return list(_PROBLEM_SPECS.keys())


def describe_problems() -> list[dict[str, Any]]:
    rows = []
    for name, problem in _PROBLEM_SPECS.items():
        rows.append(
            {
                "name": problem.name,
                "display_name": problem.display_name,
                "metric": problem.metric_name,
                "direction": problem.direction,
                "tags": ", ".join(problem.tags),
                "search_space_size": len(problem.search_space),
                "description": problem.description,
            }
        )
    return rows
