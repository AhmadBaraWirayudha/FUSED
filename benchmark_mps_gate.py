"""
benchmark_mps_gate.py
Fair comparison: MPSGate vs a dense linear gate (same architecture as
core.py's actual moe.W_gate: Matrix.he_init(d_model, num_experts) + softmax),
same synthetic task, same data, same training budget. Not wired into
FractalBrain -- this only compares the two gate architectures directly.

Task: fit a fixed random "teacher" mapping from context vectors to
expert-selection logits (unknown to either gate), train/test split, MSE.
"""
import math
import random
import sys

sys.path.insert(0, ".")
from fractal_brain.math_utils import Matrix, Vector
from fractal_brain.mps_gate import MPSGate


class DenseGate:
    """Same architecture as moe.py's W_gate (Matrix.he_init(d_model, num_experts)),
    standalone here so it can be trained on the same synthetic task as MPSGate."""
    def __init__(self, d_model, num_experts, seed=None):
        if seed is not None:
            random.seed(seed)
        self.W = Matrix.he_init(d_model, num_experts)
        self.d_model, self.num_experts = d_model, num_experts

    def num_params(self):
        return self.d_model * self.num_experts

    def forward(self, x):
        logits = self.W.linear(Vector(x)).to_list()
        return logits, x

    def backward(self, d_logits, x):
        return [[x[i] * d_logits[j] for j in range(self.num_experts)] for i in range(self.d_model)]

    def sgd_update(self, grad, lr):
        for i in range(self.d_model):
            for j in range(self.num_experts):
                self.W.data[i][j] -= lr * grad[i][j]


def make_teacher(d_model, num_experts, seed):
    random.seed(seed)
    W1 = [[random.gauss(0, 0.5) for _ in range(d_model)] for _ in range(num_experts)]
    b1 = [random.gauss(0, 0.2) for _ in range(num_experts)]

    def teacher(x):
        return [math.tanh(sum(W1[e][i] * x[i] for i in range(d_model)) + b1[e]) for e in range(num_experts)]
    return teacher


def make_dataset(d_model, teacher, n, seed):
    random.seed(seed)
    xs = [[random.gauss(0, 1) for _ in range(d_model)] for _ in range(n)]
    ys = [teacher(x) for x in xs]
    return xs, ys


def mse(pred, target):
    diff = [pred[i] - target[i] for i in range(len(pred))]
    return sum(d * d for d in diff) / len(diff), [2 * d / len(diff) for d in diff]


def train_and_eval(gate, train_x, train_y, test_x, test_y, epochs=30, lr=0.05):
    for _ in range(epochs):
        for x, y in zip(train_x, train_y):
            out = gate.forward(x)
            pred, cache = out if isinstance(out, tuple) else (out, x)
            _, d_pred = mse(pred, y)
            grad = gate.backward(d_pred, cache)
            gate.sgd_update(grad, lr)
    test_losses = []
    for x, y in zip(test_x, test_y):
        pred, _ = gate.forward(x)
        l, _ = mse(pred, y)
        test_losses.append(l)
    return sum(test_losses) / len(test_losses)


if __name__ == "__main__":
    d_model, num_experts = 32, 6
    teacher = make_teacher(d_model, num_experts, seed=1)
    train_x, train_y = make_dataset(d_model, teacher, n=200, seed=2)
    test_x, test_y = make_dataset(d_model, teacher, n=60, seed=3)

    print(f"{'gate':<12}{'params':>10}{'test_mse':>12}")
    for name, gate in [
        ("dense", DenseGate(d_model, num_experts, seed=10)),
        ("mps", MPSGate(d_model, num_experts, n_sites=8, bond_dim=6, phys_dim=4, seed=10)),
    ]:
        test_mse = train_and_eval(gate, train_x, train_y, test_x, test_y, epochs=30, lr=0.05)
        print(f"{name:<12}{gate.num_params():>10}{test_mse:>12.5f}")
