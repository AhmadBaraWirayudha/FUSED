import numpy as np
import pytest
from sklearn.datasets import load_diabetes

from metatune.benchmarks.synthetic_data import (
    METHODS,
    bootstrap_jitter,
    compare_distributions,
    gaussian_copula_like,
    gaussian_mixture_like,
    generate_like,
    kde_like,
)


@pytest.fixture(scope="module")
def real_data():
    d = load_diabetes()
    return d.data, d.target


@pytest.mark.parametrize("method", sorted(METHODS))
def test_each_method_produces_correct_shape(real_data, method):
    X, y = real_data
    synth_X, synth_y = generate_like(method, X, y, n_samples=200, seed=42)
    assert synth_X.shape == (200, X.shape[1])
    assert synth_y.shape == (200,)
    assert np.isfinite(synth_X).all()
    assert np.isfinite(synth_y).all()


@pytest.mark.parametrize("method", sorted(METHODS))
def test_each_method_is_deterministic_given_a_seed(real_data, method):
    X, y = real_data
    a_X, a_y = generate_like(method, X, y, n_samples=50, seed=7)
    b_X, b_y = generate_like(method, X, y, n_samples=50, seed=7)
    assert np.allclose(a_X, b_X)
    assert np.allclose(a_y, b_y)


@pytest.mark.parametrize("method", sorted(METHODS))
def test_each_method_reasonably_resembles_the_real_data(real_data, method):
    """Not just 'it ran' -- checks the actual resemblance, with a threshold loose enough
    to allow for each method's real, different trade-offs (see synthetic_data.py's
    docstring) but tight enough to catch a real regression, like the bandwidth bug this
    was written to catch: kde_like's correlation-frobenius-diff was 3.96 before the fix
    (a single bandwidth applied to columns on wildly different scales), 1.37 after.
    """
    X, y = real_data
    synth_X, synth_y = generate_like(method, X, y, n_samples=len(X), seed=42)
    stats = compare_distributions(X, y, synth_X, synth_y)
    assert stats["mean_abs_diff_avg"] < 2.0
    assert stats["std_abs_diff_avg"] < 2.0
    assert stats["correlation_frobenius_diff"] < 2.0


def test_unknown_method_raises(real_data):
    X, y = real_data
    with pytest.raises(ValueError):
        generate_like("not_a_real_method", X, y, n_samples=10, seed=0)


def test_gmm_and_kde_fall_back_to_gaussian_without_sklearn(monkeypatch, real_data):
    """gaussian_mixture_like / kde_like should degrade gracefully, not crash, if
    scikit-learn isn't importable -- simulated here rather than assumed.
    """
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("sklearn"):
            raise ImportError(f"simulated: {name} not installed")
        return real_import(name, *args, **kwargs)

    X, y = real_data
    monkeypatch.setattr(builtins, "__import__", fake_import)
    gmm_X, gmm_y = gaussian_mixture_like(X, y, n_samples=30, seed=1)
    kde_X, kde_y = kde_like(X, y, n_samples=30, seed=1)
    monkeypatch.undo()

    # both should match what gaussian_copula_like (numpy-only) produces directly
    fallback_X, fallback_y = gaussian_copula_like(X, y, n_samples=30, seed=1)
    assert np.allclose(gmm_X, fallback_X)
    assert np.allclose(kde_X, fallback_X)


def test_bootstrap_jitter_stays_close_to_real_marginal_ranges(real_data):
    """bootstrap_jitter resamples real rows and adds small jitter -- values should stay
    in the neighborhood of the real data's own range, not drift arbitrarily far.
    """
    X, y = real_data
    synth_X, synth_y = bootstrap_jitter(X, y, n_samples=500, seed=3, jitter_scale=0.15)
    real_y_range = y.max() - y.min()
    assert synth_y.min() > y.min() - 0.5 * real_y_range
    assert synth_y.max() < y.max() + 0.5 * real_y_range
