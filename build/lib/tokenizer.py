from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable


class SentenceSplitter:
    _pattern = re.compile(r"(?<=[.!?])\s+|\n+")

    @staticmethod
    def split(text: str) -> list[str]:
        parts = [p.strip() for p in SentenceSplitter._pattern.split(text) if p.strip()]
        return parts or ([text.strip()] if text.strip() else [])


@dataclass
class TokenBatch:
    sentences: list[str]
    token_ids: list[int]
    attention_mask: list[int]
    embeddings: list[list[float]]
    raw_tokens: list[str]


class QueryTokenizer:
    def __init__(self, tokenizer_name: str, max_length: int = 512) -> None:
        self.backend_name = tokenizer_name
        self.backend = None
        self.max_length = max_length

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> 'QueryTokenizer':
        name = config.get('model', {}).get('fallback_model_name') or config.get('model', {}).get('model_name') or 'fallback-tokenizer'
        return cls(name)

    def tokenize(self, text: str) -> TokenBatch:
        sentences = SentenceSplitter.split(text)
        raw_tokens = re.findall(r'[A-Za-z0-9_]+|[^\w\s]', text)[: self.max_length]
        token_ids = [self._stable_id(tok) for tok in raw_tokens]
        attention_mask = [1] * len(token_ids)
        embeddings = self._id_embeddings(token_ids)
        return TokenBatch(sentences, token_ids, attention_mask, embeddings, raw_tokens)

    @staticmethod
    def _stable_id(token: str) -> int:
        digest = hashlib.sha256(token.encode('utf-8')).digest()
        return int.from_bytes(digest[:4], 'big') % 32000

    @staticmethod
    def _id_embeddings(token_ids: list[int], dim: int = 64) -> list[list[float]]:
        vectors: list[list[float]] = []
        for token_id in token_ids:
            # Deterministic pseudo-embedding from token id.
            x = token_id + 13579
            row = []
            for i in range(dim):
                x = (1103515245 * x + 12345 + i) & 0x7FFFFFFF
                row.append(((x % 2000) / 1000.0) - 1.0)
            vectors.append(row)
        return vectors


class TextEmbedder:
    """Configurable text embedder with a zero-dependency hash fallback.

    ``simple-hash-embedding`` preserves the original deterministic behavior.
    ``sentence-transformers/<model>`` enables real semantic embeddings when
    the optional Sentence Transformers dependency and model are available.
    """

    SEMANTIC_PREFIX = 'sentence-transformers/'

    def __init__(
        self,
        model_name: str,
        cache_size: int = 2048,
        device: str = 'auto',
        normalize_embeddings: bool = True,
    ) -> None:
        self.model_name = model_name
        self.cache_size = cache_size
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self._cache: dict[str, list[float]] = {}
        self._model: Any = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> 'TextEmbedder':
        retrieval = config.get('retrieval', {})
        return cls(
            retrieval.get('embedding_model', 'simple-hash-embedding'),
            cache_size=int(retrieval.get('embedding_cache_size', 2048)),
            device=str(retrieval.get('embedding_device', 'auto')),
            normalize_embeddings=bool(retrieval.get('normalize_embeddings', True)),
        )

    @property
    def is_semantic(self) -> bool:
        return self.model_name.startswith(self.SEMANTIC_PREFIX)

    @property
    def backend_name(self) -> str:
        return 'sentence-transformers' if self.is_semantic else 'simple-hash'

    @property
    def dimension(self) -> int | None:
        if self._cache:
            return len(next(iter(self._cache.values())))
        if self.is_semantic:
            self._load_semantic_model()
            return int(self._model.get_sentence_embedding_dimension())
        return 384

    def embed_text(self, text: str) -> list[float]:
        if text in self._cache:
            return self._cache[text]
        if self.is_semantic:
            vec = self._semantic_encode([text])[0]
        else:
            vec = self._fallback(text)
        if len(self._cache) >= self.cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[text] = vec
        return vec

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        values = list(texts)
        missing = [text for text in values if text not in self._cache]
        if missing and self.is_semantic:
            vectors = self._semantic_encode(missing)
            for text, vec in zip(missing, vectors):
                if len(self._cache) >= self.cache_size:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[text] = vec
        return [self.embed_text(text) for text in values]

    def _load_semantic_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                'Semantic embeddings require the optional dependency '
                "'sentence-transformers'. Install with `pip install \".[semantic]\"`."
            ) from exc
        model_id = self.model_name[len(self.SEMANTIC_PREFIX):]
        kwargs: dict[str, Any] = {}
        if self.device != 'auto':
            kwargs['device'] = self.device
        self._model = SentenceTransformer(model_id, **kwargs)

    def _semantic_encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._load_semantic_model()
        encoded = self._model.encode(
            texts,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in row] for row in encoded]

    @staticmethod
    def _fallback(text: str, dim: int = 384) -> list[float]:
        vec = [0.0] * dim
        for tok in re.findall(r'[A-Za-z0-9_]+', text.lower()):
            idx = _stable_hash(tok) % dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def _stable_hash(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode('utf-8')).digest()[:8], 'big')
