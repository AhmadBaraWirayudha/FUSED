# Getting Started

This guide assumes you know little or nothing about Python.

## 1. What FUSED is

FUSED is an experimental AI pipeline. It combines memory/retrieval, planning, a custom FractalBrain controller, language-model backends, and supervised feedback.

It is a research system, not a drop-in ChatGPT replacement.

## 2. Easiest Windows path

1. Extract the repository.
2. Double-click `START_FUSED_UI.bat`.
3. Let first-run setup create the `.venv` and install dependencies.
4. The browser UI should open.

If setup fails, run `SETUP_FUSED.bat` once and start again.

## 3. First five minutes

Use the UI in this order:

```text
Start Here
   ↓
Ask
   ↓
Teach
   ↓
Benchmark
```

### Ask

Type a question and read the answer. Technical evidence is available under advanced sections.

### Teach

Enter a question and its correct answer. FUSED stores the correction as question-linked memory.

### Benchmark

Run the fixed evaluation or learning evaluation before attempting the large corpus.

## 4. If you prefer the CLI

```bash
python hybrid_cli.py --mode pipeline --text "Solve the integral of 2x from 0 to 4."
python hybrid_cli.py --mode evaluate
python hybrid_cli.py --mode evaluate-learning
```

## 5. Using the 100M-token corpus

Do not start here. First prove the small workflow works.

The reference corpus is a separate `.jsonl.gz` artifact. Once it is available:

```text
Data → detect → audit → small ingest → inspect index → benchmark → scale up
```

For a new corpus, use `GENERATE_FUSED_100M.bat` or the Data page.

## 6. Ingesting data

The accepted supervised formats are `.jsonl` and `.jsonl.gz`.

```bash
python hybrid_cli.py --mode ingest-supervised \
  --dataset path/to/data.jsonl.gz \
  --db data/supervised.db \
  --batch-size 500
```

FUSED streams the file, embeds in batches, writes SQLite in batches, and builds the retrieval index as part of the ingestion workflow.

## 7. Checking retrieval

```bash
python hybrid_cli.py --mode index-status --config config.yaml --db data/supervised.db
```

If the index is missing or stale:

```bash
python hybrid_cli.py --mode rebuild-index --config config.yaml --db data/supervised.db
```

## 8. Semantic search

The base system works without third-party embedding packages. For semantic embeddings:

```bash
pip install -e ".[semantic]"
```

For FAISS as well:

```bash
pip install -e ".[semantic,faiss]"
```

See [`SEMANTIC_RAG.md`](SEMANTIC_RAG.md).

## 9. Gemini

Gemini is optional. Put the key in an environment variable, never in source code:

```text
GEMINI_API_KEY
```

Then set `model.backend: gemini` in the active configuration.

## 10. One-click Windows files

| File | Purpose |
|---|---|
| `START_FUSED_UI.bat` | Setup if necessary and start the browser UI |
| `FUSED_ONE_CLICK.bat` | Interactive menu |
| `RUN_FUSED_ASK.bat` | Ask one question |
| `RUN_FUSED_EVALUATION.bat` | Run fixed evaluation |
| `RUN_FUSED_LEARNING.bat` | Run learning evaluation |
| `GENERATE_FUSED_100M.bat` | Generate 100M-token stress data |
| `INGEST_FUSED_DATASET.bat` | Ingest a dataset |
| `FUSED_INDEX_STATUS.bat` | Inspect retrieval index |
| `REBUILD_FUSED_INDEX.bat` | Rebuild retrieval index |
| `CHECK_FUSED_READY.bat` | Check installation readiness |
| `RUN_FUSED_RELEASE_CHECK.bat` | Local release gate |
| `OPEN_FUSED_TUTORIAL.bat` | Open this guide |

## 11. If something fails

### Python not found

Install Python 3.10+ and enable PATH integration.

### FAISS not found

FAISS is optional. The project has a non-FAISS retrieval path.

### Semantic model fails

Confirm the optional dependency is installed and that the model can be downloaded/cached on the machine.

### Gemini fails

Check that `GEMINI_API_KEY` is set and that the selected model/backend configuration is valid.

### The 100M dataset is slow

That is expected to be hardware-dependent. Start with 5,000–20,000 records and inspect throughput, memory, and retrieval latency before scaling further.

## 12. Where to read next

- Architecture → [`ARCHITECTURE.md`](ARCHITECTURE.md)
- Benchmarking → [`DATA_AND_BENCHMARKS.md`](DATA_AND_BENCHMARKS.md)
- Deployment → [`DEPLOYMENT.md`](DEPLOYMENT.md)
