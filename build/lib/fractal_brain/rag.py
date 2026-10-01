"""
fractal_brain/rag.py
In‑memory vector index, retrieval, and state‑RAG fusion with cross‑attention.
Pure Python, uses math_utils only.
"""
import math
from .math_utils import Vector, Matrix, softmax


def _mean_vector(vectors):
    n = len(vectors)
    dim = len(vectors[0])
    return [sum(v[d] for v in vectors) / n for d in range(dim)]


def _covariance_matrix(vectors, mean):
    """Sample covariance (N-1 denominator) of a list of equal-length vectors."""
    n = len(vectors)
    dim = len(mean)
    centered = [[v[d] - mean[d] for d in range(dim)] for v in vectors]
    denom = max(n - 1, 1)
    cov = [[0.0] * dim for _ in range(dim)]
    for i in range(dim):
        for j in range(i, dim):
            s = sum(centered[k][i] * centered[k][j] for k in range(n)) / denom
            cov[i][j] = s
            cov[j][i] = s
    return cov


def _cholesky(A, jitter=1e-6, max_attempts=6):
    """Cholesky decomposition A = L L^T, with escalating diagonal jitter on
    failure (the same numerical-stability approach used for metatune's GP
    kernel matrix -- see metatune/core.py's _jittered_cho_factor). Needed here
    because a document covariance matrix from a handful of embeddings is
    frequently near-singular (more embedding dimensions than documents, or
    correlated dimensions), which a plain Cholesky would reject outright.
    """
    n = len(A)
    current_jitter = max(jitter, 1e-10)
    for _ in range(max_attempts):
        L = [[0.0] * n for _ in range(n)]
        ok = True
        for i in range(n):
            for j in range(i + 1):
                s = sum(L[i][k] * L[j][k] for k in range(j))
                if i == j:
                    val = A[i][i] + current_jitter - s
                    if val <= 0.0:
                        ok = False
                        break
                    L[i][j] = math.sqrt(val)
                else:
                    L[i][j] = (A[i][j] - s) / L[j][j]
            if not ok:
                break
        if ok:
            return L
        current_jitter *= 10.0
    # last resort: heavily regularized diagonal-only fallback, still returns a valid L
    return [[math.sqrt(A[i][i] + current_jitter) if i == j else 0.0 for j in range(n)] for i in range(n)]


def _forward_substitution(L, b):
    n = len(L)
    x = [0.0] * n
    for i in range(n):
        s = sum(L[i][j] * x[j] for j in range(i))
        x[i] = (b[i] - s) / L[i][i]
    return x


class VectorStore:
    """
    A simple brute‑force vector database using dot‑product similarity.
    """
    def __init__(self, dim):
        self.dim = dim
        self.vectors = []   # list of Vector
        self.doc_ids = []   # list of document identifiers (int or str)
        self._pmi_prior = {}  # doc index -> running EMA of p(doc | past queries), for search_pmi

    def add(self, vector, doc_id):
        """Add a vector with an associated document id."""
        if not isinstance(vector, Vector):
            vector = Vector(vector)
        assert len(vector) == self.dim
        self.vectors.append(vector)
        self.doc_ids.append(doc_id)

    def search(self, query, k=5):
        """
        Return the top‑k document ids and their similarity scores.
        query: Vector (or list) of length dim.
        """
        if not isinstance(query, Vector):
            query = Vector(query)
        scores = []
        for i, vec in enumerate(self.vectors):
            # dot product similarity
            sim = query.dot(vec)
            scores.append((sim, i))
        # sort descending by similarity
        scores.sort(reverse=True, key=lambda x: x[0])
        top_k = scores[:k]
        doc_ids = [self.doc_ids[i] for _, i in top_k]
        sims = [s for s, _ in top_k]
        return doc_ids, sims

    def search_mahalanobis(self, query, k=5):
        """
        Rank documents by Mahalanobis distance to the query, using the sample
        covariance of the stored document vectors as the metric, instead of
        this store's default raw (unnormalized) dot product. A direction the
        corpus varies little along counts a given deviation as more unusual
        (larger distance) than the same deviation along a direction the corpus
        already varies a lot along -- unlike dot product or cosine similarity,
        which treat every embedding dimension as equally scaled.

        Requires at least 2 stored vectors (need something to estimate a
        covariance from). Returns (doc_ids, distances) with distances ascending
        (smaller = closer, unlike search()'s descending similarity).
        """
        if not isinstance(query, Vector):
            query = Vector(query)
        n = len(self.vectors)
        if n < 2:
            raise ValueError("search_mahalanobis needs at least 2 stored vectors to estimate a covariance")

        raw_vectors = [v.to_list() for v in self.vectors]
        mean = _mean_vector(raw_vectors)
        cov = _covariance_matrix(raw_vectors, mean)
        L = _cholesky(cov)

        q = query.to_list()
        distances = []
        for i, v in enumerate(raw_vectors):
            diff = [v[d] - q[d] for d in range(self.dim)]
            # d_M(q, v)^2 = diff^T Cov^-1 diff = ||L^-1 diff||^2 (forward substitution solves L y = diff)
            y = _forward_substitution(L, diff)
            dist_sq = sum(yi * yi for yi in y)
            distances.append((math.sqrt(max(dist_sq, 0.0)), i))
        distances.sort(key=lambda x: x[0])  # ascending: smaller distance = more similar
        top_k = distances[:k]
        doc_ids = [self.doc_ids[i] for _, i in top_k]
        dists = [d for d, _ in top_k]
        return doc_ids, dists

    def search_pmi(self, query, k=5, prior_decay=0.98):
        """
        Rank documents by an approximation of pointwise mutual information,
        Score(doc | query) = log p(doc | query) - log p(doc), rather than
        similarity alone -- so a document that scores moderately well on THIS
        query but is only ever moderately relevant in general outranks one
        that scores slightly higher here but is (per this store's own
        retrieval history) something that scores fairly well against nearly
        every query, i.e. generically similar rather than specifically
        relevant.

        p(doc | query) is approximated as the softmax of this store's existing
        dot-product similarity scores over the current candidate set.
        p(doc) is approximated as a running EMA (decay=prior_decay) of each
        document's own past p(doc | query) values across every previous call
        to this method -- an empirical marginal, not a fixed property of the
        vector alone (e.g. not just its norm), rebuilt from actual query
        traffic. A document with no search history yet starts at a uniform
        prior (1 / number of documents).

        This does maintain state (self._pmi_prior) that changes on every
        call, unlike search()/search_mahalanobis() which are pure functions of
        the current store contents -- calling it repeatedly with unrelated
        queries is what lets the prior become meaningful.
        """
        if not isinstance(query, Vector):
            query = Vector(query)
        n = len(self.vectors)
        if n == 0:
            return [], []

        sims = [query.dot(vec) for vec in self.vectors]
        posterior = softmax(Vector(sims)).to_list()  # p(doc | query) over the whole store

        uniform = 1.0 / n
        scores = []
        for i in range(n):
            prior = self._pmi_prior.get(i, uniform)
            pmi = math.log(max(posterior[i], 1e-12)) - math.log(max(prior, 1e-12))
            scores.append((pmi, i))
            self._pmi_prior[i] = prior_decay * prior + (1.0 - prior_decay) * posterior[i]

        scores.sort(reverse=True, key=lambda x: x[0])
        top_k = scores[:k]
        doc_ids = [self.doc_ids[i] for _, i in top_k]
        pmis = [s for s, _ in top_k]
        return doc_ids, pmis

    def get_vector(self, doc_id):
        """Retrieve the stored vector for a document id."""
        # find index by doc_id
        for i, did in enumerate(self.doc_ids):
            if did == doc_id:
                return self.vectors[i]
        return None


class StateRAGFusion:
    """
    Fuses a state embedding with retrieved document embeddings using
    scaled dot‑product cross‑attention (query attends to documents).
    """
    def __init__(self, d_model):
        self.d_model = d_model
        # learnable projections
        self.W_q = Matrix.he_init(d_model, d_model)   # query projection
        self.W_k = Matrix.he_init(d_model, d_model)   # key projection
        self.W_v = Matrix.he_init(d_model, d_model)   # value projection
        self.W_o = Matrix.he_init(d_model, d_model)   # output projection

    def forward(self, state_emb, retrieved_embs):
        """
        state_emb: list or Vector of length d_model
        retrieved_embs: list of lists (or Vectors), each length d_model
        Returns: fused vector (list of length d_model)
        """
        if not isinstance(state_emb, Vector):
            state_emb = Vector(state_emb)
        num_docs = len(retrieved_embs)

        # Convert retrieved embs to list of Vectors
        docs = []
        for e in retrieved_embs:
            if not isinstance(e, Vector):
                docs.append(Vector(e))
            else:
                docs.append(e)

        # Project state (query) into Q
        # state_emb is (d_model,) so treat as row vector; Q = state_emb @ W_q -> Vector
        Q = self.W_q.linear(state_emb)  # Vector (d_model,)

        # Project each document into K and V
        K = Matrix([self.W_k.linear(doc).to_list() for doc in docs])  # (num_docs, d_model)
        V = Matrix([self.W_v.linear(doc).to_list() for doc in docs])  # (num_docs, d_model)

        # Scaled dot‑product attention: scores = Q @ K^T / sqrt(d_model)
        # Q is Vector, K is Matrix; compute dot product between Q and each row of K
        scores = []
        for i in range(num_docs):
            k_vec = Vector(K.data[i])
            dot = Q.dot(k_vec)
            scores.append(dot / math.sqrt(self.d_model))
        # softmax over scores
        attn_weights = softmax(Vector(scores))  # Vector length num_docs

        # Weighted sum of V
        fused = Vector.zeros(self.d_model)
        for i in range(num_docs):
            weight = attn_weights[i]
            v_vec = Vector(V.data[i])
            fused = Vector([fused[j] + weight * v_vec[j] for j in range(self.d_model)])

        # Output projection
        out = self.W_o.linear(fused)  # Vector (d_model,)
        return out.to_list()