from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from memory import VectorMemoryStore
from tokenizer import TextEmbedder


class FakeSemanticModel:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            value = text.lower()
            if 'pump' in value:
                vectors.append([1.0, 0.0, 0.0])
            elif 'voltage' in value or 'ohm' in value:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors


def test_semantic_embedder_uses_sentence_transformer_model(monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeSemanticModel()
    class Module:
        SentenceTransformer = lambda self, model_id, **kwargs: model
    monkeypatch.setitem(__import__('sys').modules, 'sentence_transformers', Module())

    embedder = TextEmbedder('sentence-transformers/test-model')
    assert embedder.is_semantic
    assert embedder.backend_name == 'sentence-transformers'
    assert embedder.embed_text('pump flow') == [1.0, 0.0, 0.0]
    assert model.calls == [['pump flow']]


def test_semantic_model_missing_dependency_has_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(__import__('sys').modules, 'sentence_transformers', None)
    embedder = TextEmbedder('sentence-transformers/test-model')
    with pytest.raises(RuntimeError, match='sentence-transformers'):
        embedder.embed_text('pump')


def test_embedding_model_change_reembeds_existing_documents(tmp_path: Path) -> None:
    db = tmp_path / 'memory.db'
    hash_embedder = TextEmbedder('simple-hash-embedding')
    store = VectorMemoryStore(db, hash_embedder)
    store.initialize()
    store.add_document('Pump head increases with speed.', hash_embedder.embed_text('Pump head increases with speed.'), {'source': 'test'}, 'doc-1')
    old_embedding = store.documents[0].embedding

    semantic = TextEmbedder('test-semantic')
    semantic.embed_texts = lambda texts: [[0.1, 0.2, 0.3] for _ in texts]  # type: ignore[method-assign]
    store2 = VectorMemoryStore(db, semantic)
    store2.initialize()

    assert store2.documents[0].embedding == [0.1, 0.2, 0.3]
    assert store2.documents[0].embedding != old_embedding
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT value FROM memory_meta WHERE key='embedding_model'").fetchone()[0] == 'test-semantic'
    conn.close()


def test_semantic_vectors_retrieve_by_cosine(tmp_path: Path) -> None:
    class FakeEmbedder:
        model_name = 'fake-semantic'
        def embed_text(self, text: str):
            return {
                'hydraulic pump': [1.0, 0.0, 0.0],
                'electric voltage': [0.0, 1.0, 0.0],
            }[text]
        def embed_texts(self, texts):
            return [self.embed_text(t) for t in texts]

    store = VectorMemoryStore(tmp_path / 'memory.db', FakeEmbedder())
    store.initialize()
    store.add_document('hydraulic pump', [1.0, 0.0, 0.0], {}, 'pump')
    store.add_document('electric voltage', [0.0, 1.0, 0.0], {}, 'voltage')
    result = store.retrieve([0.99, 0.01, 0.0], top_k=2)
    assert result[0].doc_id == 'pump'
    assert result[0].score > result[1].score
