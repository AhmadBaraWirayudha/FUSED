from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class MemoryDocument:
    doc_id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any]
    success_count: int = 0
    created_at: float = 0.0
    updated_at: float = 0.0
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class InteractionRecord:
    interaction_id: str
    input_text: str
    retrieved_doc_ids: list[str]
    plan: dict[str, Any]
    final_output: str
    success: Optional[bool]
    metadata: dict[str, Any]


class VectorMemoryStore:
    def __init__(
        self,
        sqlite_path: str | Path,
        embedder: Any,
        backend: str = 'auto',
        max_documents: int = 10000,
        base_path: str | Path | None = None,
        sqlite_wal: bool = True,
        sqlite_busy_timeout_ms: int = 5000,
        write_batch_size: int = 500,
        persistent_index_dir: str | Path = 'data/retrieval_index',
        persistent_index: bool = True,
        index_chunk_size: int = 8192,
    ) -> None:
        self.base_path = Path(base_path) if base_path is not None else Path.cwd()
        self.sqlite_path = self._resolve_path(sqlite_path)
        self.embedder = embedder
        self.backend = backend
        self.max_documents = max_documents
        self.conn: sqlite3.Connection | None = None
        self.documents: list[MemoryDocument] = []
        self.interactions: dict[str, InteractionRecord] = {}
        self._faiss_index = None
        self._faiss_ids: list[int] = []
        self._embedding_signature: str | None = None
        self.sqlite_wal = bool(sqlite_wal)
        self.sqlite_busy_timeout_ms = max(100, int(sqlite_busy_timeout_ms))
        self.write_batch_size = max(1, int(write_batch_size))
        self.persistent_index_dir = self._resolve_path(persistent_index_dir)
        self.persistent_index = bool(persistent_index)
        self.index_chunk_size = max(256, int(index_chunk_size))
        self._persistent_backend: str | None = None
        self._persistent_vectors = None
        self._persistent_doc_ids: list[str] = []
        self._persistent_index_valid = False
        self._persistent_manifest: dict[str, Any] = {}

    @classmethod
    def from_config(cls, config: dict[str, Any], embedder: Any, base_path: str | Path | None = None) -> 'VectorMemoryStore':
        return cls(
            config['paths']['sqlite_db'],
            embedder,
            config.get('retrieval', {}).get('backend', 'auto'),
            int(config.get('retrieval', {}).get('max_documents', 10000)),
            base_path=base_path,
            sqlite_wal=bool(config.get('runtime', {}).get('sqlite_wal', True)),
            sqlite_busy_timeout_ms=int(config.get('runtime', {}).get('sqlite_busy_timeout_ms', 5000)),
            write_batch_size=int(config.get('runtime', {}).get('memory_write_batch_size', 500)),
            persistent_index_dir=config.get('paths', {}).get('retrieval_index_dir', 'data/retrieval_index'),
            persistent_index=bool(config.get('retrieval', {}).get('persistent_index', True)),
            index_chunk_size=int(config.get('retrieval', {}).get('index_chunk_size', 8192)),
        )

    def initialize(self) -> None:
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.sqlite_path, timeout=self.sqlite_busy_timeout_ms / 1000.0)
        self.conn.execute(f'PRAGMA busy_timeout={self.sqlite_busy_timeout_ms}')
        if self.sqlite_wal:
            self.conn.execute('PRAGMA journal_mode=WAL')
        self.conn.execute('PRAGMA synchronous=NORMAL')
        self.conn.execute(
            '''CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                embedding TEXT NOT NULL,
                metadata TEXT NOT NULL,
                success_count INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )'''
        )
        self.conn.execute(
            '''CREATE TABLE IF NOT EXISTS interactions (
                interaction_id TEXT PRIMARY KEY,
                input_text TEXT NOT NULL,
                retrieved_doc_ids TEXT NOT NULL,
                plan TEXT NOT NULL,
                final_output TEXT NOT NULL,
                success INTEGER,
                metadata TEXT NOT NULL,
                corrected_output TEXT,
                notes TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )'''
        )
        self.conn.execute("CREATE TABLE IF NOT EXISTS memory_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.conn.commit()
        self.load_documents()
        self._ensure_embedding_compatibility()
        if not self._load_persistent_index():
            self._rebuild_persistent_index() if self.persistent_index else None
        if self._persistent_backend != 'faiss':
            self._rebuild_optional_index()

    def _resolve_path(self, path: str | Path) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        return (self.base_path / p).resolve()

    def load_bootstrap_records(self, dataset_path: Path) -> list[dict[str, Any]]:
        dataset_path = self._resolve_path(dataset_path)
        return [
            json.loads(line)
            for line in dataset_path.read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]

    def bootstrap(self, records: list[dict[str, Any]]) -> None:
        rows: list[tuple[str, str, dict[str, Any]]] = []
        for r in records:
            text = r.get('solution') or r['query']
            meta = dict(r.get('metadata', {}))
            meta.update(
                {
                    'problem_type': r.get('problem_type'),
                    'query': r.get('query'),
                    'actions': r.get('actions', []),
                    'step_texts': r.get('step_texts', []),
                    'bootstrap': True,
                }
            )
            doc_id = 'bootstrap:' + _stable_bootstrap_id(r.get('query', ''), text)
            rows.append((doc_id, text, meta))
        self.bulk_add_documents(rows)

    def load_documents(self) -> list[MemoryDocument]:
        if self.conn is None:
            return []
        rows = self.conn.execute(
            'SELECT doc_id, text, embedding, metadata, success_count, created_at, updated_at FROM documents'
        ).fetchall()
        self.documents = [
            MemoryDocument(
                row[0],
                row[1],
                json.loads(row[2]),
                json.loads(row[3]),
                int(row[4]),
                float(row[5]),
                float(row[6]),
            )
            for row in rows
        ][: self.max_documents]
        return self.documents

    def count_documents(self) -> int:
        if self.conn is None:
            raise RuntimeError('Memory store not initialized')
        row = self.conn.execute('SELECT COUNT(*) FROM documents').fetchone()
        return int(row[0]) if row else 0

    def bulk_add_documents(
        self,
        records: list[tuple[str, str, dict[str, Any]]],
        batch_size: int | None = None,
        rebuild_index: bool = True,
    ) -> int:
        """Insert many documents in batches and rebuild the vector index once.

        Each record is ``(doc_id, text, metadata)``. Embeddings are generated
        in batches from the store's embedder, which avoids the old
        ``add_document`` -> commit -> FAISS rebuild loop for every row.
        """
        if self.conn is None:
            raise RuntimeError('Memory store not initialized')
        if not records:
            return 0
        size = max(1, int(batch_size or self.write_batch_size))
        inserted = 0
        now = time.time()
        for start in range(0, len(records), size):
            chunk = records[start:start + size]
            texts = [item[1] for item in chunk]
            embeddings = self.embedder.embed_texts(texts)
            if len(embeddings) != len(chunk):
                raise RuntimeError('Batch embedding returned a mismatched document count')
            rows = []
            for (doc_id, text, metadata), embedding in zip(chunk, embeddings):
                emb = [float(x) for x in embedding]
                rows.append((doc_id, text, json.dumps(emb), json.dumps(metadata), 0, now, now))
            before = self.conn.total_changes
            self.conn.executemany(
                'INSERT OR IGNORE INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)',
                rows,
            )
            inserted += self.conn.total_changes - before
            self.conn.commit()

        if rebuild_index and inserted > 0:
            self.load_documents()
            if self.persistent_index:
                self._rebuild_persistent_index()
            if not self.persistent_index or self._persistent_backend != 'faiss':
                self._rebuild_optional_index()
        elif inserted == 0:
            # Idempotent bootstrap: do not rebuild a valid persistent index
            # just because the same bootstrap records were presented again.
            pass
        else:
            self._invalidate_persistent_index()
        return int(inserted)

    def bulk_add_learning_examples(
        self,
        examples: list[tuple[str, str, dict[str, Any] | None]],
        batch_size: int | None = None,
        rebuild_index: bool = True,
    ) -> int:
        """Bulk-ingest supervised question/answer lessons with deterministic ids."""
        records: list[tuple[str, str, dict[str, Any]]] = []
        for question, answer, metadata in examples:
            q = question.strip()
            a = answer.strip()
            if not q or not a:
                raise ValueError('question and answer must not be empty')
            meta = dict(metadata or {})
            meta.update({'source': meta.get('source', 'supervised_dataset'), 'query': q, 'answer': a})
            doc_id = 'learning:' + _stable_learning_id(q, a)
            records.append((doc_id, a, meta))
        return self.bulk_add_documents(records, batch_size=batch_size, rebuild_index=rebuild_index)

    def add_document(self, text: str, embedding: list[float], metadata: dict[str, Any], doc_id: str | None = None) -> str:
        if self.conn is None:
            raise RuntimeError('Memory store not initialized')
        doc_id = doc_id or str(uuid.uuid4())
        now = time.time()
        emb = [float(x) for x in embedding]
        cur = self.conn.execute(
            'INSERT OR IGNORE INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)',
            (doc_id, text, json.dumps(emb), json.dumps(metadata), 0, now, now),
        )
        self.conn.commit()
        if cur.rowcount == 0:
            # A document with this id was already present (e.g. bootstrap
            # running again on an already-loaded dataset). Skip adding it to
            # the in-memory index too, so we don't accumulate duplicates
            # there either. See CHANGELOG.
            return doc_id
        self.documents.append(MemoryDocument(doc_id, text, emb, metadata, 0, now, now))
        self.documents = self.documents[-self.max_documents :]
        self._invalidate_persistent_index()
        self._rebuild_optional_index()
        return doc_id


    def add_learning_example(
        self,
        question: str,
        answer: str,
        metadata: dict[str, Any] | None = None,
        doc_id: str | None = None,
    ) -> str:
        """Store a question->answer lesson so retrieval matches the question.

        The answer remains the document payload shown to the generator, while
        the embedding is built from both the question and answer. This avoids
        the previous failure mode where a corrected answer such as ``16`` was
        embedded without the question that led to it and therefore was hard
        to retrieve on the next turn.
        """
        question = question.strip()
        answer = answer.strip()
        if not question:
            raise ValueError('question must not be empty')
        if not answer:
            raise ValueError('answer must not be empty')
        payload = dict(metadata or {})
        payload.update({'source': payload.get('source', 'learning'), 'query': question, 'answer': answer})
        embedding = self.embedder.embed_text(f'Question: {question}\nAnswer: {answer}')
        return self.add_document(answer, embedding, payload, doc_id=doc_id)

    def retrieve(self, query_embedding: list[float], top_k: int = 3) -> list[MemoryDocument]:
        if top_k <= 0 or self.count_documents() <= 0:
            return []

        if self.persistent_index:
            self._ensure_persistent_index()
            if self._persistent_backend == 'faiss' and self._faiss_index is not None:
                return self._retrieve_faiss_persistent(query_embedding, top_k)
            if self._persistent_backend == 'numpy' and self._persistent_vectors is not None:
                return self._retrieve_numpy_persistent(query_embedding, top_k)

        if not self.documents:
            self.load_documents()
        top_k = min(int(top_k), len(self.documents))
        if self._faiss_index is not None:
            try:
                import numpy as np

                query = np.asarray([query_embedding], dtype='float32')
                scores, indices = self._faiss_index.search(query, top_k)
                ranked = [
                    (float(score), int(index))
                    for score, index in zip(scores[0], indices[0])
                    if int(index) >= 0
                ]
            except Exception:
                ranked = []
        else:
            scores = [cosine_similarity(d.embedding, query_embedding) for d in self.documents]
            ranked = sorted(
                ((float(score), idx) for idx, score in enumerate(scores)),
                key=lambda item: item[0],
                reverse=True,
            )[:top_k]

        out: list[MemoryDocument] = []
        for score, idx in ranked[:top_k]:
            d = self.documents[idx]
            out.append(
                MemoryDocument(
                    d.doc_id, d.text, d.embedding, d.metadata, d.success_count,
                    d.created_at, d.updated_at, score,
                )
            )
        return out

    def _retrieve_faiss_persistent(self, query_embedding: list[float], top_k: int) -> list[MemoryDocument]:
        import numpy as np

        top_k = min(int(top_k), len(self._persistent_doc_ids))
        query = np.asarray([query_embedding], dtype='float32')
        faiss = __import__('faiss')
        faiss.normalize_L2(query)
        scores, indices = self._faiss_index.search(query, top_k)
        ranked = [
            (float(score), self._persistent_doc_ids[int(index)])
            for score, index in zip(scores[0], indices[0])
            if int(index) >= 0
        ]
        return self._materialize_ranked_documents(ranked)

    def _retrieve_numpy_persistent(self, query_embedding: list[float], top_k: int) -> list[MemoryDocument]:
        import numpy as np

        query = np.asarray(query_embedding, dtype='float32')
        q_norm = float(np.linalg.norm(query)) or 1.0
        query = query / q_norm
        candidates: list[tuple[float, int]] = []
        total = len(self._persistent_doc_ids)
        for start in range(0, total, self.index_chunk_size):
            end = min(start + self.index_chunk_size, total)
            scores = np.asarray(self._persistent_vectors[start:end] @ query, dtype='float32')
            local_k = min(int(top_k), len(scores))
            if local_k <= 0:
                continue
            idxs = np.argpartition(scores, -local_k)[-local_k:]
            candidates.extend((float(scores[i]), start + int(i)) for i in idxs)
        candidates.sort(key=lambda item: item[0], reverse=True)
        ranked = [(score, self._persistent_doc_ids[idx]) for score, idx in candidates[:top_k]]
        return self._materialize_ranked_documents(ranked)

    def _materialize_ranked_documents(self, ranked: list[tuple[float, str]]) -> list[MemoryDocument]:
        if self.conn is None or not ranked:
            return []
        ids = [doc_id for _, doc_id in ranked]
        placeholders = ','.join('?' for _ in ids)
        rows = self.conn.execute(
            f'SELECT doc_id, text, embedding, metadata, success_count, created_at, updated_at FROM documents WHERE doc_id IN ({placeholders})',
            ids,
        ).fetchall()
        by_id = {
            row[0]: MemoryDocument(
                row[0], row[1], json.loads(row[2]), json.loads(row[3]),
                int(row[4]), float(row[5]), float(row[6]),
            )
            for row in rows
        }
        output: list[MemoryDocument] = []
        for score, doc_id in ranked:
            doc = by_id.get(doc_id)
            if doc is not None:
                doc.score = score
                output.append(doc)
        return output

    def store_pending_interaction(self, record: InteractionRecord) -> None:
        if self.conn is None:
            raise RuntimeError('Memory store not initialized')
        now = time.time()
        self.conn.execute(
            'INSERT OR REPLACE INTO interactions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM interactions WHERE interaction_id=?), ?), ?)',
            (
                record.interaction_id,
                record.input_text,
                json.dumps(record.retrieved_doc_ids),
                json.dumps(record.plan),
                record.final_output,
                None if record.success is None else int(record.success),
                json.dumps(record.metadata),
                None,
                None,
                record.interaction_id,
                now,
                now,
            ),
        )
        self.conn.commit()
        self.interactions[record.interaction_id] = record

    def finalize_interaction(self, interaction_id: str, success: bool, corrected_output: str | None = None, notes: str | None = None) -> dict[str, Any]:
        if self.conn is None:
            raise RuntimeError('Memory store not initialized')
        now = time.time()
        self.conn.execute(
            'UPDATE interactions SET success=?, corrected_output=?, notes=?, updated_at=? WHERE interaction_id=?',
            (int(success), corrected_output, notes, now, interaction_id),
        )
        self.conn.commit()
        return {'interaction_id': interaction_id, 'success': success, 'corrected_output': corrected_output, 'notes': notes}

    def _ensure_embedding_compatibility(self) -> None:
        """Record embedder identity and migrate every stored vector after a model change."""
        if self.conn is None:
            raise RuntimeError('Memory store not initialized')
        signature = str(getattr(self.embedder, 'model_name', type(self.embedder).__name__))
        row = self.conn.execute('SELECT value FROM memory_meta WHERE key=?', ('embedding_model',)).fetchone()
        previous = row[0] if row else None
        if previous == signature:
            self._embedding_signature = signature
            return

        count = self.count_documents()
        if count:
            batch_rows = []
            offset = 0
            while True:
                rows = self.conn.execute(
                    'SELECT doc_id, text FROM documents ORDER BY rowid LIMIT ? OFFSET ?',
                    (self.index_chunk_size, offset),
                ).fetchall()
                if not rows:
                    break
                embeddings = self.embedder.embed_texts([row[1] for row in rows])
                if len(embeddings) != len(rows):
                    raise RuntimeError('Embedding migration returned a mismatched document count')
                now = time.time()
                batch_rows = [(json.dumps([float(v) for v in emb]), now, doc_id) for (doc_id, _), emb in zip(rows, embeddings)]
                self.conn.executemany(
                    'UPDATE documents SET embedding=?, updated_at=? WHERE doc_id=?',
                    batch_rows,
                )
                self.conn.commit()
                offset += len(rows)
                if len(rows) < self.index_chunk_size:
                    break

        self.conn.execute(
            'INSERT OR REPLACE INTO memory_meta(key, value) VALUES(?, ?)',
            ('embedding_model', signature),
        )
        self.conn.commit()
        self._embedding_signature = signature
        self._invalidate_persistent_index()

    def rebuild_index(self) -> None:
        """Rebuild the persistent retrieval index and the active in-memory/FAISS cache."""
        self.load_documents()
        if self.persistent_index:
            self._rebuild_persistent_index()
        if not self.persistent_index or self._persistent_backend != 'faiss':
            self._rebuild_optional_index()

    def close(self) -> None:
        if self._persistent_vectors is not None:
            try:
                self._persistent_vectors._mmap.close()
            except Exception:
                pass
        self._persistent_vectors = None
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def _db_fingerprint(self) -> tuple[int, float, float]:
        if self.conn is None:
            return (0, 0.0, 0.0)
        row = self.conn.execute(
            'SELECT COUNT(*), COALESCE(MIN(updated_at), 0), COALESCE(MAX(updated_at), 0) FROM documents'
        ).fetchone()
        return (int(row[0]), float(row[1]), float(row[2])) if row else (0, 0.0, 0.0)

    def _index_manifest_path(self) -> Path:
        return self.persistent_index_dir / 'manifest.json'

    def _invalidate_persistent_index(self) -> None:
        self._persistent_index_valid = False
        self._persistent_vectors = None
        self._persistent_doc_ids = []
        self._persistent_backend = None
        self._persistent_manifest = {}

    def _ensure_persistent_index(self) -> None:
        if not self.persistent_index:
            return
        if self._persistent_index_valid:
            return
        if not self._load_persistent_index():
            self._rebuild_persistent_index()

    def _load_persistent_index(self) -> bool:
        if not self.persistent_index or self.conn is None:
            return False
        manifest_path = self._index_manifest_path()
        if not manifest_path.exists():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            db_count, db_min, db_max = self._db_fingerprint()
            count_match = manifest.get('document_count') == db_count
            time_match = (abs(float(manifest.get('min_updated_at', 0)) - db_min) < 1e-9 and
                          abs(float(manifest.get('max_updated_at', 0)) - db_max) < 1e-9)
            model_match = manifest.get('embedding_model') == self._embedding_signature
            if not (count_match and time_match and model_match):
                return False

            backend = str(manifest.get('backend', 'numpy'))
            doc_ids_path = self.persistent_index_dir / manifest['doc_ids_file']
            doc_ids = json.loads(doc_ids_path.read_text(encoding='utf-8'))
            if len(doc_ids) != db_count:
                return False

            if backend == 'faiss':
                import faiss
                import numpy as np  # noqa: F401
                self._faiss_index = faiss.read_index(str(self.persistent_index_dir / manifest['index_file']))
                self._persistent_vectors = None
            elif backend == 'numpy':
                import numpy as np
                dtype = np.dtype(manifest.get('dtype', 'float32'))
                shape = tuple(manifest['shape'])
                self._persistent_vectors = np.memmap(
                    self.persistent_index_dir / manifest['index_file'],
                    dtype=dtype,
                    mode='r',
                    shape=shape,
                )
                self._faiss_index = None
            else:
                return False

            self._persistent_backend = backend
            self._persistent_doc_ids = [str(x) for x in doc_ids]
            self._persistent_manifest = manifest
            self._persistent_index_valid = True
            return True
        except Exception:
            self._invalidate_persistent_index()
            return False

    def _rebuild_persistent_index(self) -> None:
        if not self.persistent_index or self.conn is None:
            return
        self.persistent_index_dir.mkdir(parents=True, exist_ok=True)
        import numpy as np

        db_count, db_min, db_max = self._db_fingerprint()
        old_files = [
            'vectors.float32.bin', 'vectors.faiss', 'doc_ids.json', 'manifest.json',
        ]
        for name in old_files:
            path = self.persistent_index_dir / name
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass

        rows = self.conn.execute(
            'SELECT doc_id, embedding FROM documents ORDER BY rowid'
        )
        first = rows.fetchone()
        if first is None:
            self._invalidate_persistent_index()
            return
        first_embedding = np.asarray(json.loads(first[1]), dtype='float32')
        dim = int(first_embedding.shape[0])
        if dim <= 0:
            raise RuntimeError('Stored embedding dimension must be positive')
        doc_ids = [str(first[0])]

        # Build FAISS when explicitly requested or available in auto mode.
        want_faiss = str(self.backend or 'auto').lower() in {'auto', 'faiss'}
        faiss = None
        if want_faiss:
            try:
                import faiss as _faiss
                faiss = _faiss
            except ImportError:
                if str(self.backend or 'auto').lower() == 'faiss':
                    raise RuntimeError(
                        'FAISS retrieval was requested but `faiss-cpu` is not installed. ' +
                        'Install with `pip install .[faiss]`.'
                    )

        if faiss is not None:
            index = faiss.IndexFlatIP(dim)
            vec = first_embedding.reshape(1, -1)
            faiss.normalize_L2(vec)
            index.add(vec)
            while True:
                chunk = rows.fetchmany(self.index_chunk_size)
                if not chunk:
                    break
                matrix = np.asarray([json.loads(row[1]) for row in chunk], dtype='float32')
                if matrix.ndim != 2 or matrix.shape[1] != dim:
                    raise RuntimeError('Stored embeddings have inconsistent dimensions')
                faiss.normalize_L2(matrix)
                index.add(matrix)
                doc_ids.extend(str(row[0]) for row in chunk)
            faiss.write_index(index, str(self.persistent_index_dir / 'vectors.faiss'))
            backend = 'faiss'
            shape = [db_count, dim]
            index_file = 'vectors.faiss'
            dtype = 'float32'
            self._faiss_index = index
            self._persistent_vectors = None
        else:
            vectors_path = self.persistent_index_dir / 'vectors.float32.bin'
            mmap = np.memmap(vectors_path, dtype='float32', mode='w+', shape=(db_count, dim))
            offset = 0
            pending = [first]
            while pending:
                matrix = np.asarray([json.loads(row[1]) for row in pending], dtype='float32')
                if matrix.ndim != 2 or matrix.shape[1] != dim:
                    raise RuntimeError('Stored embeddings have inconsistent dimensions')
                norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                mmap[offset:offset + len(pending)] = matrix / norms
                offset += len(pending)
                pending = rows.fetchmany(self.index_chunk_size)
                if pending:
                    doc_ids.extend(str(row[0]) for row in pending)
            if offset != db_count:
                raise RuntimeError(f'Persistent index row mismatch: wrote {offset}, expected {db_count}')
            mmap.flush()
            del mmap
            backend = 'numpy'
            shape = [db_count, dim]
            index_file = 'vectors.float32.bin'
            dtype = 'float32'
            self._persistent_vectors = np.memmap(
                vectors_path, dtype='float32', mode='r', shape=(db_count, dim)
            )
            self._faiss_index = None

        (self.persistent_index_dir / 'doc_ids.json').write_text(
            json.dumps(doc_ids, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
        )
        manifest = {
            'schema_version': 1,
            'backend': backend,
            'index_file': index_file,
            'doc_ids_file': 'doc_ids.json',
            'dtype': dtype,
            'shape': shape,
            'document_count': db_count,
            'embedding_dimension': dim,
            'embedding_model': self._embedding_signature,
            'min_updated_at': db_min,
            'max_updated_at': db_max,
            'index_chunk_size': self.index_chunk_size,
        }
        manifest_path = self._index_manifest_path()
        tmp = manifest_path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        tmp.replace(manifest_path)
        self._persistent_backend = backend
        self._persistent_doc_ids = doc_ids
        self._persistent_manifest = manifest
        self._persistent_index_valid = True

    def _rebuild_optional_index(self) -> None:
        self._faiss_index = None
        self._faiss_ids = []
        backend = str(self.backend or 'auto').lower()
        if backend not in {'auto', 'faiss'} or not self.documents:
            return
        try:
            import faiss
            import numpy as np
        except ImportError:
            if backend == 'faiss':
                raise RuntimeError(
                    'FAISS retrieval was requested but `faiss-cpu` is not installed. '
                    'Install with `pip install .[faiss]`.'
                )
            return

        vectors = np.asarray([d.embedding for d in self.documents], dtype='float32')
        if vectors.ndim != 2 or vectors.shape[1] == 0:
            return
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        self._faiss_index = index
        self._faiss_ids = list(range(len(self.documents)))


def _stable_learning_id(question: str, answer: str) -> str:
    digest = hashlib.sha256(f'{question}\x1f{answer}'.encode('utf-8')).hexdigest()
    return digest[:24]


def _stable_bootstrap_id(query: str, text: str) -> str:
    digest = hashlib.sha256(f'{query}\x1f{text}'.encode('utf-8')).hexdigest()
    return digest[:24]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    length = min(len(a), len(b))
    dot = sum(float(a[i]) * float(b[i]) for i in range(length))
    na = math.sqrt(sum(float(x) * float(x) for x in a[:length])) or 1.0
    nb = math.sqrt(sum(float(x) * float(x) for x in b[:length])) or 1.0
    return dot / (na * nb)
