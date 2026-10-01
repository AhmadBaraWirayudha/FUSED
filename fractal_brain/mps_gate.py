"""Experimental MPS-style gate. Dense gate remained faster in the benchmark; kept for research reference, not as the active default."""

"""
fractal_brain/mps_gate.py
A Matrix Product State (MPS) encoder as an alternative expert-selection gate,
standalone from core.py's dense W_gate so both can be benchmarked fairly (see
benchmark_mps_gate.py). Not wired into FractalBrain -- this is the "genuinely
new, separate component" from the tensor-network audit, not a claim that the
existing gate already is one.

Contraction: input x (d_model,) is split into n_sites chunks; each chunk is
locally projected + tanh'd into a phys_dim feature vector; a chain of 3-index
site tensors (bond_left, phys, bond_right) absorbs each site's feature vector
in sequence, propagating a bond_dim-wide "bond vector" left to right (site 0
has a fixed length-1 left boundary). The final bond vector is read out via a
linear layer to num_experts logits. This is the standard way tensor-network
sequence models (e.g. Stoudenmire & Schwab's tensor-network classifier) turn
a chain of site tensors into a function of a fixed-size input -- not a novel
scheme invented here.
"""
import math
import random


def _mat_vec(rows, cols, M, v):
    return [sum(M[i][j] * v[j] for j in range(cols)) for i in range(rows)]


class MPSGate:
    def __init__(self, d_model, num_experts, n_sites=8, bond_dim=6, phys_dim=4, seed=None):
        assert d_model % n_sites == 0, "d_model must divide evenly into n_sites"
        if seed is not None:
            random.seed(seed)
        self.d_model, self.num_experts = d_model, num_experts
        self.n_sites, self.bond_dim, self.phys_dim = n_sites, bond_dim, phys_dim
        self.chunk = d_model // n_sites

        def he(fan_in):
            return math.sqrt(2.0 / max(fan_in, 1))

        # per-site local projection: chunk -> phys_dim
        self.W_local = [[[random.gauss(0, he(self.chunk)) for _ in range(self.chunk)] for _ in range(phys_dim)]
                         for _ in range(n_sites)]
        self.b_local = [[0.0] * phys_dim for _ in range(n_sites)]
        # site tensors: site 0 has bond_left=1; all others bond_left=bond_dim
        self.T = [[[[random.gauss(0, he(bond_dim * phys_dim)) for _ in range(bond_dim)] for _ in range(phys_dim)]
                   for _ in range(1 if s == 0 else bond_dim)] for s in range(n_sites)]
        # readout: bond_dim -> num_experts
        self.W_out = [[random.gauss(0, he(bond_dim)) for _ in range(bond_dim)] for _ in range(num_experts)]
        self.b_out = [0.0] * num_experts

    def num_params(self):
        n = sum(self.phys_dim * self.chunk + self.phys_dim for _ in range(self.n_sites))
        n += sum((1 if s == 0 else self.bond_dim) * self.phys_dim * self.bond_dim for s in range(self.n_sites))
        n += self.num_experts * self.bond_dim + self.num_experts
        return n

    def forward(self, x):
        """Returns (logits, cache) where cache holds everything backward() needs."""
        phys_vecs, bonds = [], []
        bond_in = [1.0]  # length-1 boundary
        for s in range(self.n_sites):
            chunk = x[s * self.chunk:(s + 1) * self.chunk]
            raw = _mat_vec(self.phys_dim, self.chunk, self.W_local[s], chunk)
            raw = [raw[p] + self.b_local[s][p] for p in range(self.phys_dim)]
            phys = [math.tanh(v) for v in raw]
            phys_vecs.append(phys)

            T_s = self.T[s]
            bond_out = [0.0] * self.bond_dim
            for bp in range(len(bond_in)):
                for p in range(self.phys_dim):
                    w = bond_in[bp] * phys[p]
                    if w == 0.0:
                        continue
                    row = T_s[bp][p]
                    for bo in range(self.bond_dim):
                        bond_out[bo] += w * row[bo]
            bond_out = [math.tanh(v) for v in bond_out]  # bound the propagated bond vector, like an RNN hidden
            # state -- an unbounded linear contraction across n_sites multiplies up over the chain and diverges
            # during training (confirmed: without this, loss reached NaN within 5 SGD steps at lr=0.05).
            bonds.append(bond_out)
            bond_in = bond_out

        logits = _mat_vec(self.num_experts, self.bond_dim, self.W_out, bond_in)
        logits = [logits[e] + self.b_out[e] for e in range(self.num_experts)]
        cache = {"x": x, "phys_vecs": phys_vecs, "bonds": bonds}
        return logits, cache

    def backward(self, d_logits, cache):
        """d_logits: dLoss/dlogits (length num_experts). Returns a grads dict
        with the same key structure as the corresponding self.* attributes,
        for use by sgd_update() below."""
        phys_vecs, bonds, x = cache["phys_vecs"], cache["bonds"], cache["x"]
        n_sites, bond_dim, phys_dim, chunk = self.n_sites, self.bond_dim, self.phys_dim, self.chunk

        d_W_out = [[d_logits[e] * bonds[-1][b] for b in range(bond_dim)] for e in range(self.num_experts)]
        d_b_out = list(d_logits)
        d_bond = [sum(d_logits[e] * self.W_out[e][b] for e in range(self.num_experts)) for b in range(bond_dim)]

        d_T = [None] * n_sites
        d_W_local = [None] * n_sites
        d_b_local = [None] * n_sites

        for s in range(n_sites - 1, -1, -1):
            bond_in = bonds[s - 1] if s > 0 else [1.0]
            phys = phys_vecs[s]
            T_s = self.T[s]
            nb_in = len(bond_in)

            bond_out_s = bonds[s]  # tanh'd; d_bond is dLoss/d(bond_out_s) -- convert to pre-tanh first
            d_bond_raw = [d_bond[bo] * (1.0 - bond_out_s[bo] * bond_out_s[bo]) for bo in range(bond_dim)]

            dT_s = [[[0.0] * bond_dim for _ in range(phys_dim)] for _ in range(nb_in)]
            d_bond_in = [0.0] * nb_in
            d_phys = [0.0] * phys_dim
            for bp in range(nb_in):
                for p in range(phys_dim):
                    w = bond_in[bp] * phys[p]
                    row = T_s[bp][p]
                    for bo in range(bond_dim):
                        g = d_bond_raw[bo]
                        dT_s[bp][p][bo] += w * g
                        d_bond_in[bp] += phys[p] * row[bo] * g
                        d_phys[p] += bond_in[bp] * row[bo] * g
            d_T[s] = dT_s

            d_raw = [d_phys[p] * (1.0 - phys[p] * phys[p]) for p in range(phys_dim)]  # tanh'
            d_b_local[s] = d_raw
            xc = x[s * chunk:(s + 1) * chunk]
            d_W_local[s] = [[d_raw[p] * xc[j] for j in range(chunk)] for p in range(phys_dim)]

            d_bond = d_bond_in  # becomes gradient w.r.t. previous site's bond_out

        return {"W_out": d_W_out, "b_out": d_b_out, "T": d_T, "W_local": d_W_local, "b_local": d_b_local}

    def sgd_update(self, grads, lr):
        for e in range(self.num_experts):
            for b in range(self.bond_dim):
                self.W_out[e][b] -= lr * grads["W_out"][e][b]
            self.b_out[e] -= lr * grads["b_out"][e]
        for s in range(self.n_sites):
            nb_in = len(self.T[s])
            for bp in range(nb_in):
                for p in range(self.phys_dim):
                    for bo in range(self.bond_dim):
                        self.T[s][bp][p][bo] -= lr * grads["T"][s][bp][p][bo]
            for p in range(self.phys_dim):
                for j in range(self.chunk):
                    self.W_local[s][p][j] -= lr * grads["W_local"][s][p][j]
                self.b_local[s][p] -= lr * grads["b_local"][s][p]
