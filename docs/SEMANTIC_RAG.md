# Semantic RAG

FUSED supports a dependency-free baseline and an optional real semantic-embedding path.

## Baseline

`config.yaml` uses:

```yaml
retrieval:
  embedding_model: simple-hash-embedding
  backend: auto
```

This keeps the core path deterministic and lightweight.

## Semantic embeddings

`config.semantic.yaml` uses a Sentence Transformer model:

```yaml
retrieval:
  embedding_model: sentence-transformers/all-MiniLM-L6-v2
  backend: auto
```

Install:

```bash
pip install -e ".[semantic]"
```

## FAISS

Install:

```bash
pip install -e ".[faiss]"
```

With normalized vectors, the FAISS inner-product search is used for cosine-style similarity. Without FAISS, FUSED retains an exact local vector-search implementation.

## Safe model changes

The memory database records the embedding-model identity. When the model changes, stored documents are re-embedded rather than mixing incompatible vectors.

## Evidence boundary

Adding semantic embeddings changes the mechanism. It does not by itself establish better answer accuracy. Run the evaluation benchmark before and after any retrieval change.
