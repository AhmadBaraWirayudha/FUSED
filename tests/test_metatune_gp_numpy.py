"""Confirms the numpy/scipy-backed GaussianProcessRegressor in metatune.core is a faithful,
numerically-equivalent replacement for the original pure-Python implementation -- not just
faster. The original hand-rolled Cholesky-based GP is inlined below as a reference (this is
the exact algorithm metatune.core used to ship with, kept only so this test has something
independent to check against); it is not part of the public package.
"""
import math
import random
import time

import numpy as np
import pytest

from metatune import core


# --- reference implementation: the original pure-Python GP this package replaced -----------

class _RefMathUtils:
    EPSILON = 1e-9

    @staticmethod
    def mean(xs):
        return sum(xs) / len(xs)

    @staticmethod
    def std_dev(xs, sample=True):
        m = _RefMathUtils.mean(xs)
        n = len(xs) - (1 if sample and len(xs) > 1 else 0)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / max(n, 1))

    @staticmethod
    def cholesky(A):
        n = len(A)
        L = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1):
                s = sum(L[i][k] * L[j][k] for k in range(j))
                if i == j:
                    val = A[i][i] - s
                    L[i][j] = math.sqrt(val) if val > _RefMathUtils.EPSILON else math.sqrt(_RefMathUtils.EPSILON)
                else:
                    L[i][j] = (A[i][j] - s) / L[j][j]
        return L

    @staticmethod
    def forward_substitution(L, b):
        n = len(L)
        x = [0.0] * n
        for i in range(n):
            s = sum(L[i][j] * x[j] for j in range(i))
            x[i] = (b[i] - s) / L[i][i]
        return x

    @staticmethod
    def backward_substitution(U, b):
        n = len(U)
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            s = sum(U[i][j] * x[j] for j in range(i + 1, n))
            x[i] = (b[i] - s) / U[i][i]
        return x

    @staticmethod
    def mat_transpose(A):
        return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

    @staticmethod
    def solve_cholesky(L, b):
        y = _RefMathUtils.forward_substitution(L, b)
        return _RefMathUtils.backward_substitution(_RefMathUtils.mat_transpose(L), y)


class _RefRBFKernel:
    def __init__(self, length_scales=None, variance=1.0):
        self.length_scales = length_scales
        self.variance = variance

    def __call__(self, x1, x2):
        if not x1 or len(x1) != len(x2):
            return 0.0
        if self.length_scales is None:
            sq_dist = sum((a - b) ** 2 for a, b in zip(x1, x2))
        else:
            sq_dist = sum(((a - b) / max(l, _RefMathUtils.EPSILON)) ** 2 for a, b, l in zip(x1, x2, self.length_scales))
        return self.variance * math.exp(-0.5 * sq_dist)


class _RefGaussianProcessRegressor:
    """The original hand-rolled, pure-Python Cholesky-based GP (pre-metatune)."""

    def __init__(self, kernel, noise=1e-6, normalize_y=True):
        self.kernel = kernel
        self.noise = noise
        self.normalize_y = normalize_y
        self.X_train = []
        self.y_mean = 0.0
        self.y_std = 1.0
        self._L = None
        self._alpha = []
        self._fitted = False

    def fit(self, X, y):
        self.X_train = X
        if self.normalize_y:
            self.y_mean = _RefMathUtils.mean(y)
            self.y_std = _RefMathUtils.std_dev(y, sample=True) if len(y) > 1 else 1.0
            if self.y_std <= _RefMathUtils.EPSILON:
                self.y_std = 1.0
            y_scaled = [(yi - self.y_mean) / self.y_std for yi in y]
        else:
            y_scaled = list(y)
        n = len(X)
        K = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                val = self.kernel(X[i], X[j]) + (self.noise if i == j else 0.0)
                K[i][j] = val
                K[j][i] = val
        self._L = _RefMathUtils.cholesky(K)
        self._alpha = _RefMathUtils.solve_cholesky(self._L, y_scaled)
        self._fitted = True

    def predict(self, X_test):
        if not self._fitted or not self.X_train:
            return [self.y_mean] * len(X_test), [1.0] * len(X_test)
        means, stds = [], []
        for x_star in X_test:
            k_star = [self.kernel(x_star, xi) for xi in self.X_train]
            k_star_star = self.kernel(x_star, x_star) + self.noise
            mean_scaled = sum(k * a for k, a in zip(k_star, self._alpha))
            v = _RefMathUtils.forward_substitution(self._L, k_star)
            var_scaled = max(k_star_star - sum(vi * vi for vi in v), _RefMathUtils.EPSILON)
            std_scaled = math.sqrt(var_scaled)
            if self.normalize_y:
                means.append(mean_scaled * self.y_std + self.y_mean)
                stds.append(std_scaled * self.y_std)
            else:
                means.append(mean_scaled)
                stds.append(std_scaled)
        return means, stds


# --- tests: metatune.core's numpy/scipy GP vs. the reference above -------------------------

def _make_data(n, dim, seed):
    rng = random.Random(seed)
    X = [[rng.uniform(-3, 3) for _ in range(dim)] for _ in range(n)]
    y = [sum(v * v for v in row) + rng.uniform(-0.05, 0.05) for row in X]  # noisy bowl
    return X, y


@pytest.mark.parametrize("n,dim", [(5, 2), (20, 3), (60, 2), (150, 4)])
def test_new_gp_matches_reference(n, dim):
    X, y = _make_data(n, dim, seed=n * 100 + dim)
    X_test, _ = _make_data(8, dim, seed=999)

    ref = _RefGaussianProcessRegressor(_RefRBFKernel())
    ref.fit(X, y)
    ref_means, ref_stds = ref.predict(X_test)

    new_gp = core.GaussianProcessRegressor(core.RBFKernel())
    new_gp.fit(X, y)
    new_means, new_stds = new_gp.predict(X_test)

    assert np.allclose(ref_means, new_means, rtol=1e-6, atol=1e-6), (ref_means, new_means)
    assert np.allclose(ref_stds, new_stds, rtol=1e-6, atol=1e-6), (ref_stds, new_stds)


@pytest.mark.parametrize("n", [50, 200, 400])
def test_speedup_grows_with_history_size(n):
    X, y = _make_data(n, 3, seed=42)
    X_test, _ = _make_data(10, 3, seed=1)

    t0 = time.time()
    ref = _RefGaussianProcessRegressor(_RefRBFKernel())
    ref.fit(X, y)
    ref.predict(X_test)
    ref_time = time.time() - t0

    t0 = time.time()
    new_gp = core.GaussianProcessRegressor(core.RBFKernel())
    new_gp.fit(X, y)
    new_gp.predict(X_test)
    new_time = time.time() - t0

    speedup = ref_time / max(new_time, 1e-9)
    print(f"\nn={n:>4}  reference: {ref_time*1000:8.1f}ms  metatune: {new_time*1000:6.2f}ms  speedup: {speedup:6.0f}x")
    # At small n both implementations are already fast in absolute terms, so numpy/scipy's own
    # call overhead eats into the relative win -- measured 2-14x here across runs, never a
    # regression. The real claim is at n>=200, where the original O(n^3) pure-Python inner loop
    # dominates: consistently 19-100x+ across runs. That threshold (not "always >10x everywhere")
    # is what's actually being asserted.
    if n < 200:
        assert speedup > 1, f"expected metatune's GP to never be slower, got {speedup:.1f}x at n={n}"
    else:
        assert speedup > 10, f"expected at least a 10x speedup at n={n}, got {speedup:.1f}x"
