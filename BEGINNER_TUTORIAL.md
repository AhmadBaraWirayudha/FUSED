# FUSED Beginner Tutorial

This guide assumes you are not a programmer.

## 1. What is FUSED?

FUSED is a Python research project that combines several small systems:

- memory/search (RAG)
- task breaking and planning
- a custom FractalBrain scoring system
- an optional Gemini language model
- supervised teaching / feedback
- a persistent retrieval index

Think of it as a small experimental "AI laboratory" rather than a finished ChatGPT replacement.

## 2. What you need

For the normal offline demo:

- Windows 10 or newer
- Python 3.10 or newer
- the FUSED folder extracted somewhere you can find it

You do **not** need Gemini, FAISS, or a GPU for the basic demo.

## 3. The easiest way to start

Open the project folder and double-click:

`FUSED_ONE_CLICK.bat`

A menu appears. Pick a number.

The safest first choices are:

1. **Show the live demo** — watch FUSED run four real examples.
2. **Ask FUSED one question** — type your own question.
3. **Teach FUSED a fact** — give it a question and the correct answer.
13. **Run all tests** — checks the project.
14. **Open the beginner tutorial** — opens this document.

## 4. Asking a question

Choose option 2.

Type something like:

`Solve the integral of 2x from 0 to 4.`

FUSED will print JSON. You do not need to understand every line.

Look for:

- `final_output` — the answer it produced
- `retrieved` — information it found in memory
- `fractal` / `route` information — how FractalBrain affected the run
- `trace` — what happened at each stage

## 5. Teaching FUSED

Choose option 3.

Example:

Question:

`What is the safe working load for a 10mm shaft bearing?`

Correct answer:

`The safe working load for a 10mm shaft bearing is 2.4 kN.`

FUSED stores the lesson as question-linked memory.

Ask the same question later and the learned lesson can be retrieved.

## 6. Running the small benchmark

Choose option 4.

This uses a tiny fixed evaluation set. It is useful for regression testing.

The benchmark is intentionally small. A result on four cases is not proof of general intelligence.

## 7. The large synthetic dataset

TB10 produced a separate stress-test dataset of about 100 million estimated tokens.

Use option 7 only when you intentionally want to create a new large corpus.

The project already has a reference 100M-token artifact, so you normally do **not** need to regenerate it.

The large file is data, not the FUSED program.

## 8. Ingesting a dataset

Choose option 8.

Enter:

1. the `.jsonl` or `.jsonl.gz` file path
2. the database path, such as `data/supervised.db`

FUSED will:

1. read records as a stream
2. make embeddings in batches
3. write records to SQLite in batches
4. build/update the persistent retrieval index once

For the supplied TB10 dataset, the file is gzip-compressed JSONL, so you do not need to manually unzip the JSONL first.

## 9. What is the retrieval index?

Without a persistent index, FUSED would have to rebuild its search structure after a restart.

TB11 adds a saved index next to the selected SQLite database.

For example:

```text
data/
├── supervised.db
└── supervised.retrieval_index/
    ├── manifest.json
    ├── doc_ids.json
    └── vectors.float32.bin
```

The exact files can differ when FAISS is installed. With FAISS, the vector file can be `vectors.faiss`.

## 10. Checking the index

Choose option 9, or run:

```bat
python hybrid_cli.py --mode index-status --config config.yaml --db data/supervised.db
```

Useful fields are:

- `document_count` — records in the database
- `persistent_index_valid` — whether the saved index matches the database
- `persistent_backend` — `numpy` or `faiss`
- `index_dir` — where the saved index lives

## 11. Rebuilding the index

Choose option 10, or run:

```bat
python hybrid_cli.py --mode rebuild-index --config config.yaml --db data/supervised.db
```

Do this after deliberately changing embeddings or recovering a damaged index.

Normally ingestion already rebuilds the index for you.

## 12. Semantic RAG

The default project uses a simple deterministic embedding so the basic demo works offline.

For real semantic embeddings, install the optional packages:

```bat
python -m pip install -e ".[semantic,faiss]" --no-build-isolation
```

Then use `config.semantic.yaml`.

This can download Python packages and model files, so internet access may be required.

## 13. Gemini

Gemini is optional.

Set:

```text
GEMINI_API_KEY=your_key_here
```

Then select a config with:

```yaml
model:
  backend: gemini
```

The offline fallback still exists so the project can be tested without an API key.

## 14. Which .bat file should I use?

| File | Purpose |
|---|---|
| `FUSED_ONE_CLICK.bat` | Main menu for almost everything |
| `RUN_FUSED.bat` | Run the default pipeline |
| `RUN_FUSED_ASK.bat` | Type one question and run it |
| `RUN_FUSED_TESTS.bat` | Run the full test command |
| `RUN_FUSED_ALL_CHECKS.bat` | Run evaluation + learning + targeted tests + 156 smoke checks |
| `RUN_FUSED_EVALUATION.bat` | Run the fixed evaluation benchmark |
| `RUN_FUSED_LEARNING.bat` | Run the supervised-learning benchmark |
| `GENERATE_FUSED_100M.bat` | Generate a new ~100M-token corpus |
| `INGEST_FUSED_DATASET.bat` | Ingest a JSONL/JSONL.GZ dataset |
| `FUSED_INDEX_STATUS.bat` | Inspect persistent-index status for the default database |
| `REBUILD_FUSED_INDEX.bat` | Rebuild the default database's index |
| `OPEN_FUSED_TUTORIAL.bat` | Open this guide |

## 15. What not to do

Do not treat the 100M-token synthetic corpus as if it were 100M tokens of real-world knowledge.

Do not put the giant dataset into normal Git history.

Do not interpret a small synthetic benchmark as proof that FUSED is generally intelligent.

Do not put a real Gemini API key into a source file.

## 16. A simple test sequence

Use this order when you are first learning the project:

```text
1. FUSED_ONE_CLICK.bat
2. Option 1 — demo
3. Option 2 — ask a question
4. Option 3 — teach a fact
5. Option 4 — evaluation
6. Option 9 — index status
7. Option 12 — targeted tests
```

After that, try the large dataset:

```text
1. Option 8 — ingest
2. Option 9 — check index
3. Ask a question
4. Restart FUSED
5. Ask another question
6. Check index again
```

The important TB11 idea is simple:

> **The database stores the memory. The persistent index stores the search structure, so FUSED does not have to rebuild that structure every time it starts.**

## 17. If something fails

### "Python was not found"

Install Python 3.10+ and enable **Add Python to PATH** during installation.

### "Module not found: faiss"

That is normal for the default setup. The project can use the NumPy persistent index instead. FAISS is optional.

### Semantic model error

Install the semantic extra and make sure the required model files can be downloaded.

### Gemini error

Check `GEMINI_API_KEY` and the selected Gemini configuration. The offline fallback backend does not require an API key.

### Large dataset is slow

The TB10 corpus is intentionally large. Use a smaller `--max-records` value first when learning the workflow.

## 18. Where the code starts

For learning the software itself, read in this order:

```text
hybrid_cli.py
    ↓
ai_pipeline.py
    ↓
engine.py
    ↓
memory.py
    ↓
fractal_router.py
    ↓
fractal_brain/core.py
```

For large-data work:

```text
large_data.py
    ↓
memory.py
    ↓
synthetic_data.py
```
