"""Synthetic data generators that aim to resemble a *real* reference dataset,
rather than the arbitrary (if configurable) structure of sklearn's
make_regression -- which is what problems.py's original synthetic_xgboost
problem uses. Each method here fits some model of the real data's
distribution, then samples from that model. All operate on a joint (X, y)
matrix so the target's relationship to the features carries over, not just
the marginal feature distribution.

Four methods, in increasing order of how much structure they capture (and
how much it costs to fit/sample):

- gaussian_copula_like: matches the real data's mean vector and full
  covariance matrix exactly (a multivariate Gaussian fit) -- numpy only,
  cheapest, but assumes linear correlation structure and unimodal, symmetric
  marginals. Good baseline; won't capture multi-modal or skewed real data.
- gaussian_mixture_like: fits a Gaussian mixture (a handful of Gaussian
  "clusters"), captures multi-modal structure a single Gaussian can't.
  Needs scikit-learn (already a benchmarks-extra dependency, nothing new).
  Falls back to gaussian_copula_like if scikit-learn isn't importable.
- kde_like: fits a kernel density estimate over the joint data and samples
  from it -- non-parametric, captures arbitrary shape without assuming a
  fixed number of modes, at the cost of being slower for larger n. Also
  scikit-learn; falls back to gaussian_copula_like.
- bootstrap_jitter: resamples real rows with replacement and adds small
  Gaussian jitter scaled to each column's own std. The simplest way to
  extend a small real dataset to a larger synthetic one while keeping every
  marginal distribution close to exact (it IS the real data, just jittered
  and resampled) -- weakest at generating genuinely novel structure, but the
  hardest method to get "unrealistic" by construction.

compare_distributions(real, synthetic) gives a concrete way to check how
close a method actually got, rather than assuming any of them worked:
per-feature mean/std absolute differences and a correlation-matrix
Frobenius-norm difference, real units, not just claims.
"""
from __future__ import annotations

from typing import Any

import numpy as np

Array = np.ndarray


def _joint(X: Array, y: Array) -> Array:
    return np.concatenate([X, y.reshape(-1, 1)], axis=1)


def _split(joint: Array) -> tuple[Array, Array]:
    return joint[:, :-1], joint[:, -1]


def gaussian_copula_like(X: Array, y: Array, n_samples: int, seed: int) -> tuple[Array, Array]:
    """Multivariate-Gaussian fit to the real joint data's mean and covariance,
    then sampled -- matches first and second moments exactly in expectation.
    """
    rng = np.random.default_rng(seed)
    joint = _joint(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64))
    mean = joint.mean(axis=0)
    cov = np.cov(joint, rowvar=False)
    cov = cov + 1e-9 * np.eye(cov.shape[0])  # jitter for numerical stability on near-singular cov
    synth = rng.multivariate_normal(mean, cov, size=n_samples)
    return _split(synth.astype(np.float32))


def gaussian_mixture_like(
    X: Array, y: Array, n_samples: int, seed: int, n_components: int = 3
) -> tuple[Array, Array]:
    """Gaussian mixture fit to the real joint data -- captures multi-modal
    structure a single Gaussian averages away. Falls back to the single-
    Gaussian method if scikit-learn isn't installed.
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        return gaussian_copula_like(X, y, n_samples, seed)

    joint = _joint(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64))
    n_components = max(1, min(n_components, len(joint) // 5 or 1))
    gmm = GaussianMixture(n_components=n_components, random_state=seed, reg_covar=1e-6)
    gmm.fit(joint)
    synth, _ = gmm.sample(n_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(synth)  # gmm.sample() groups samples by component; shuffle so trial-order isn't biased
    return _split(synth.astype(np.float32))


def kde_like(X: Array, y: Array, n_samples: int, seed: int, bandwidth: float | str = "scott") -> tuple[Array, Array]:
    """Kernel density estimate over the real joint data, sampled from --
    non-parametric, no fixed number of modes assumed. Falls back to the
    single-Gaussian method if scikit-learn isn't installed.

    Standardizes each column before fitting and de-standardizes after
    sampling: a single bandwidth applies isotropically in KDE, so columns on
    very different scales (e.g. diabetes: X columns std ~0.05, y std ~77)
    would otherwise force one bandwidth to be simultaneously way too wide
    for X and unusably narrow for y. Verified this was a real failure mode,
    not a hypothetical one -- see the comparison numbers in
    tests/test_synthetic_data.py.
    """
    try:
        from sklearn.neighbors import KernelDensity
    except ImportError:
        return gaussian_copula_like(X, y, n_samples, seed)

    joint = _joint(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64))
    col_mean = joint.mean(axis=0)
    col_std = np.where(joint.std(axis=0) > 1e-12, joint.std(axis=0), 1.0)
    standardized = (joint - col_mean) / col_std

    if isinstance(bandwidth, str):
        # Scott's rule, a standard KDE bandwidth heuristic: n^(-1/(d+4)); safe to apply a
        # single scalar bandwidth now that every column has unit variance.
        n, d = standardized.shape
        bandwidth = float(n ** (-1.0 / (d + 4)))
    kde = KernelDensity(bandwidth=max(bandwidth, 1e-6), kernel="gaussian")
    kde.fit(standardized)
    synth_standardized = kde.sample(n_samples, random_state=seed)
    synth = synth_standardized * col_std + col_mean
    return _split(synth.astype(np.float32))


def bootstrap_jitter(X: Array, y: Array, n_samples: int, seed: int, jitter_scale: float = 0.15) -> tuple[Array, Array]:
    """Resample real rows with replacement, then add Gaussian jitter scaled
    to jitter_scale * each column's own std -- stays close to the real
    marginal distributions by construction, at the cost of limited novelty
    (it's the real data, resampled and perturbed, not a fitted model of it).
    """
    rng = np.random.default_rng(seed)
    joint = _joint(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64))
    idx = rng.integers(0, len(joint), size=n_samples)
    resampled = joint[idx].copy()
    col_std = joint.std(axis=0)
    jitter = rng.normal(0.0, jitter_scale * col_std, size=resampled.shape)
    synth = resampled + jitter
    return _split(synth.astype(np.float32))


METHODS = {
    "gaussian": gaussian_copula_like,
    "gmm": gaussian_mixture_like,
    "kde": kde_like,
    "bootstrap": bootstrap_jitter,
}


def generate_like(method: str, X: Array, y: Array, n_samples: int, seed: int, **kwargs: Any) -> tuple[Array, Array]:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(METHODS)}")
    return METHODS[method](X, y, n_samples, seed, **kwargs)


def compare_distributions(real_X: Array, real_y: Array, synth_X: Array, synth_y: Array) -> dict[str, Any]:
    """How close did it actually get? Real numbers, not a claim: per-feature
    mean/std absolute differences, plus a correlation-matrix Frobenius-norm
    difference (0 = identical linear correlation structure).
    """
    real_joint = _joint(np.asarray(real_X, dtype=np.float64), np.asarray(real_y, dtype=np.float64))
    synth_joint = _joint(np.asarray(synth_X, dtype=np.float64), np.asarray(synth_y, dtype=np.float64))

    mean_abs_diff = np.abs(real_joint.mean(axis=0) - synth_joint.mean(axis=0))
    std_abs_diff = np.abs(real_joint.std(axis=0) - synth_joint.std(axis=0))

    real_corr = np.corrcoef(real_joint, rowvar=False)
    synth_corr = np.corrcoef(synth_joint, rowvar=False)
    corr_frobenius_diff = float(np.linalg.norm(real_corr - synth_corr))

    return {
        "mean_abs_diff_per_column": mean_abs_diff.tolist(),
        "mean_abs_diff_avg": float(mean_abs_diff.mean()),
        "std_abs_diff_per_column": std_abs_diff.tolist(),
        "std_abs_diff_avg": float(std_abs_diff.mean()),
        "correlation_frobenius_diff": corr_frobenius_diff,
    }
