# FUSED

**Experimental hybrid AI system combining semantic RAG, persistent memory, FractalBrain routing, supervised feedback learning, configurable LLM backends, and a beginner-friendly web UI.**

FUSED is a Python research project for experimenting with a modular AI pipeline rather than relying on a single monolithic model.

The system combines:

* Semantic retrieval / RAG
* Persistent vector memory
* Task decomposition and planning
* FractalBrain confidence-based routing
* Supervised feedback learning
* Configurable language-model backends
* Living Knowledge Graph (`lkg.py`)
* Adaptive optimization and MetaTune
* Large synthetic-data generation and ingestion
* Reproducible evaluation and smoke testing
* Streamlit browser UI for non-programmers

> **Research status:** experimental. The project is suitable for research, benchmarking, prototyping, and portfolio demonstration. It is not presented as a production-grade general AI system.

---

## What FUSED Does

At a high level:

```text
User input
    ↓
Normalize
    ↓
Retrieve relevant memory
    ↓
Decompose the task
    ↓
Build a plan
    ↓
Estimate confidence with FractalBrain
    ↓
Select a generation policy
    ↓
Generate an answer
    ↓
Reflect on the result
    ↓
Learn from explicit corrections
```

The `LivingKnowledgeGraph` tracks task intent and document reliability across a session and can influence later retrieval and generation.

---

# Quick Start

## Option 1 — Windows, no Python knowledge required

Extract the repository and double-click:

```text
START_FUSED_UI.bat
```

The launcher will:

1. Check for the FUSED virtual environment.
2. Run setup when required.
3. Install the UI dependencies.
4. Start the Streamlit application.
5. Open the browser interface.

The browser UI is organized into:

```text
Start Here
Ask
Teach
Data
Benchmark
System
Deploy
```

For a complete beginner, start with:

```text
Start Here → Ask → Teach → Benchmark
```

---

## Option 2 — Python CLI

From the repository root:

```bash
python hybrid_cli.py --mode pipeline
```

Run a multi-turn session:

```bash
python hybrid_cli.py --mode session \
  --text "First question || Second question || Third question"
```

Run the legacy subsystems:

```bash
python hybrid_cli.py --mode closed-loop
python hybrid_cli.py --mode fractal
python -m fractal_brain --demo
```

---

# Web UI

The browser interface is located at:

```text
app/streamlit_app.py
```

Run it manually:

```bash
streamlit run app/streamlit_app.py
```

The UI is designed to hide technical details unless they are needed.

## Start Here

Shows whether the local installation is ready and provides the main actions.

## Ask

Ask FUSED a question and inspect:

* final answer
* retrieved evidence
* intent
* FractalBrain route

## Teach

Provide:

```text
Question
Correct answer
```

FUSED stores the correction as supervised memory that can be retrieved later.

## Data

Manage:

* dataset audit
* supervised-data generation
* dataset ingestion
* the 100M-token stress corpus

## Benchmark

Run:

* fixed evaluation
* learning evaluation
* data/ingestion benchmarks

Results are presented in the browser and can be saved for further analysis.

## System

Inspect the retrieval database and persistent index.

## Deploy

Provides the local release checklist and deployment information.

---

# Architecture

FUSED is organized as several cooperating layers.

```text
                         FUSED
                           │
        ┌──────────────────┼──────────────────┐
        ↓                  ↓                  ↓
      Memory           Reasoning          Language
        │                  │                  │
   Semantic RAG       FractalBrain        LLM backend
   Persistent DB      Task planner        Gemini
   Vector index       Confidence          Fallback
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ↓
                     Final response
                           ↓
                     Reflection
                           ↓
                   Supervised learning
```

### Main components

| Component                | Purpose                                  |
| ------------------------ | ---------------------------------------- |
| `hybrid_cli.py`          | Main CLI entry point                     |
| `ai_pipeline.py`         | Unified pipeline orchestrator            |
| `engine.py`              | Closed-loop execution                    |
| `memory.py`              | Retrieval and persistent memory          |
| `tokenizer.py`           | Embedding/tokenization utilities         |
| `decomposer.py`          | Task decomposition                       |
| `planner.py`             | Planning                                 |
| `decoder.py`             | Final answer generation                  |
| `language_model.py`      | Provider-neutral LLM interface           |
| `gemini_backend.py`      | Gemini backend                           |
| `fractal_router.py`      | Confidence-based routing                 |
| `fractal_brain/`         | Experimental FractalBrain implementation |
| `lkg.py`                 | Living Knowledge Graph                   |
| `learning_evaluation.py` | Supervised learning evaluation           |
| `synthetic_data.py`      | Controlled synthetic-data generation     |
| `large_data.py`          | Large supervised-data ingestion          |
| `adaptive_optimizer.py`  | Adaptive configuration optimizer         |
| `pipeline_optimizer.py`  | Optimizer integration                    |
| `metatune/`              | Optimization framework                   |
| `evaluation.py`          | Reproducible evaluation                  |
| `app/`                   | Streamlit UI                             |

---

# Language Model Backends

FUSED uses a provider-neutral interface:

```text
language_model.py
        ↓
LanguageModelBackend
        ↓
┌───────────────────────┐
│                       │
Fallback              Gemini
offline               API
```

The fallback backend keeps the core project usable without external credentials.

Gemini can be enabled through configuration:

```yaml
model:
  backend: gemini
  model_name: gemini-3.8-flash
  gemini:
    api_key_env: GEMINI_API_KEY
    thinking_level: medium
```

Set the API key before running:

### Windows Command Prompt

```bat
set GEMINI_API_KEY=YOUR_API_KEY
```

### PowerShell

```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY"
```

FUSED does not require a Gemini key for the offline fallback backend or the core regression/smoke checks.

---

# Semantic RAG

FUSED supports two retrieval approaches.

## Dependency-free default

```text
simple hash embedding
        ↓
cosine similarity
```

This keeps the base installation lightweight.

## Semantic mode

Optional Sentence Transformer embeddings can be enabled with:

```bash
pip install -e ".[semantic]"
```

FAISS can additionally be installed with:

```bash
pip install -e ".[faiss]"
```

The semantic path is:

```text
Question
    ↓
Embedding model
    ↓
Vector search
    ↓
Relevant memory
    ↓
LLM / fallback generator
```

Existing documents are re-embedded when the embedding model changes so different vector spaces are not mixed.

---

# Persistent Retrieval Index

Large supervised databases can maintain a persistent retrieval index.

Example:

```text
data/
├── supervised.db
└── supervised.retrieval_index/
    ├── manifest.json
    ├── doc_ids.json
    └── vectors.float32.bin
```

Inspect an index:

```bash
python hybrid_cli.py \
  --mode index-status \
  --config config.yaml \
  --db data/supervised.db
```

Rebuild it manually:

```bash
python hybrid_cli.py \
  --mode rebuild-index \
  --config config.yaml \
  --db data/supervised.db
```

The index is reused when its database fingerprint and embedding model match.

If it is stale or missing, FUSED rebuilds it.

---

# FractalBrain Routing

FractalBrain produces a pre-generation confidence estimate.

That estimate selects one of three policies:

```text
High confidence
    ↓
direct_grounded

Medium confidence
    ↓
standard

Low confidence
    ↓
cautious_reasoning
```

The route can affect generation behavior such as:

* token budget
* reasoning level
* cautious-generation behavior

See:

```text
docs/FRACTAL_ROUTING.md
```

This means FractalBrain is not only producing a score for display; its output can change downstream behavior.

---

# Supervised Learning

FUSED can learn from an explicit correction:

```text
Question
    ↓
Incorrect answer
    ↓
Human provides correction
    ↓
Question + correction stored
    ↓
Future retrieval
    ↓
Corrected answer becomes available
```

Run the learning benchmark:

```bash
python hybrid_cli.py --mode evaluate-learning
```

The learning evaluation measures whether a supervised correction becomes retrievable and improves the tested response.

---

# Synthetic Supervised Data

FUSED can generate controlled supervised datasets from its benchmark seeds.

Example:

```bash
python hybrid_cli.py \
  --mode generate-supervised \
  --dataset data/synthetic_supervised_10000.jsonl \
  --count 10000
```

Audit a generated dataset:

```bash
python synthetic_data.py audit \
  --dataset data/synthetic_supervised_10000.jsonl
```

The generated records contain provenance metadata and use an explicit train/evaluation split.

The synthetic datasets are intended for:

* stress testing
* retrieval testing
* supervised-ingestion testing
* benchmark development
* reproducibility experiments

They are **not** treated as proof of real-world knowledge quality.

---

# 100M-Token Stress Corpus

TB10 introduced a large synthetic stress corpus.

The delivered dataset contains approximately:

```text
179,619 records
99,999,890 estimated tokens
80% train
20% evaluation
0 exact train/evaluation input overlap
100% provenance completeness
```

The corpus is intentionally distributed separately from the normal source repository.

## Generate a 100M-token corpus yourself

```bash
python hybrid_cli.py \
  --mode generate-max-supervised \
  --dataset data/tb10/synthetic_supervised_max.jsonl.gz \
  --token-budget 100000000 \
  --target-tokens-per-record 512 \
  --seed 20261001 \
  --train-ratio 0.8
```

## Windows one-click option

Use:

```text
FUSED_ONE_CLICK.bat
```

or the browser UI:

```text
Data → 100M-token data
```

The UI can:

* detect the corpus
* audit it
* generate it
* ingest it

Large operations require explicit confirmation.

The 100M-token corpus should **not** be committed to normal Git history.

---

# Large-Data Ingestion

Supervised JSONL/JSONL.GZ data can be streamed into FUSED.

Example:

```bash
python hybrid_cli.py \
  --mode ingest-supervised \
  --dataset data/synthetic_supervised_10000.jsonl \
  --db data/supervised.db \
  --batch-size 500
```

The ingestion system supports:

* streaming reads
* batch embeddings
* batch SQLite writes
* persistent indexes
* provenance-aware supervised records

The current implementation is intended for research-scale testing rather than unlimited distributed production search.

---

# Evaluation

FUSED has a versioned evaluation contract.

Run the standard evaluation:

```bash
python hybrid_cli.py --mode evaluate
```

The benchmark reports:

* answer accuracy
* retrieval Hit@1 / Hit@K
* MRR
* latency
* per-case results
* retrieved evidence

Evaluation data is stored under:

```text
data/evaluation/
```

The project also contains learning and large-data benchmark tooling.

---

# Benchmarking and Optimization

FUSED contains two optimization layers.

## Adaptive optimizer

```text
adaptive_optimizer.py
pipeline_optimizer.py
```

These can search tunable configuration values such as:

* retrieval threshold
* planner size
* knowledge-graph settings
* other pipeline parameters

Run:

```bash
python hybrid_cli.py --mode tune
```

## MetaTune

`metatune/` provides an Optuna-shaped API:

```python
from metatune import Study, Float, Int, Categorical
```

The project can use accelerated NumPy/SciPy implementations when installed:

```bash
pip install -e ".[fast]"
```

Benchmark-related dependencies are available through:

```bash
pip install -e ".[benchmarks]"
```

See:

```text
docs/BENCHMARKS.md
docs/ADAPTIVE_OPTIMIZER.md
```

---

# Installation

## Core installation

The base package is intentionally lightweight.

```bash
python -m venv .venv
```

### Windows

```bat
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Then:

```bash
python -m pip install --upgrade pip
pip install -e .
```

Optional acceleration:

```bash
pip install -e ".[fast]"
```

Semantic RAG:

```bash
pip install -e ".[semantic]"
```

FAISS:

```bash
pip install -e ".[faiss]"
```

Benchmark stack:

```bash
pip install -e ".[benchmarks]"
```

---

# One-Click Windows Files

For non-programmers, use the provided `.bat` files.

| File                          | Purpose                                     |
| ----------------------------- | ------------------------------------------- |
| `START_FUSED_UI.bat`          | Install if needed and start browser UI      |
| `SETUP_FUSED.bat`             | Create environment and install dependencies |
| `RUN_FUSED_UI.bat`            | Start the UI using the existing environment |
| `CHECK_FUSED_READY.bat`       | Check whether the installation is ready     |
| `FUSED_ONE_CLICK.bat`         | Main interactive command menu               |
| `RUN_FUSED_RELEASE_CHECK.bat` | Run the local release validation            |

For a complete beginner:

```text
START_FUSED_UI.bat
```

is the preferred entry point.

---

# Testing

Run the targeted suite:

```bash
pytest
```

Run the standalone FractalBrain smoke suite:

```bash
python fractal_brain/tests/smoke_tests.py
```

The project also contains:

```bash
python run_tests.py
```

for the broader test workflow.

The current project includes a dedicated smoke-test bridge so the standalone 156-check suite can be exercised through normal pytest infrastructure.

---

# Project Structure

```text
FUSED/
├── app/
│   ├── streamlit_app.py
│   └── README.md
│
├── fractal_brain/
│   ├── core.py
│   ├── attention.py
│   ├── rag.py
│   ├── tokenizer.py
│   └── ...
│
├── metatune/
│   └── ...
│
├── tests/
│   └── ...
│
├── docs/
│   ├── BEGINNER_TUTORIAL.md
│   ├── UNIFIED_PIPELINE.md
│   ├── FRACTAL_ROUTING.md
│   ├── SYNTHETIC_DATA.md
│   ├── BENCHMARKS.md
│   ├── ADAPTIVE_OPTIMIZER.md
│   └── ...
│
├── data/
│   ├── bootstrap_dataset.jsonl
│   └── evaluation/
│
├── hybrid_cli.py
├── ai_pipeline.py
├── engine.py
├── memory.py
├── tokenizer.py
├── decomposer.py
├── planner.py
├── decoder.py
├── language_model.py
├── gemini_backend.py
├── fractal_router.py
├── large_data.py
├── synthetic_data.py
├── evaluation.py
├── learning_evaluation.py
├── adaptive_optimizer.py
├── pipeline_optimizer.py
│
├── config.yaml
├── config.semantic.yaml
├── pyproject.toml
├── Dockerfile
├── LICENSE
├── .gitignore
├── README.md
│
└── Windows launchers
```

---

# Deployment

## Streamlit

The main browser entry point is:

```text
app/streamlit_app.py
```

Dependency file:

```text
app/requirements.txt
```

The project also contains:

```text
.python-version
Dockerfile
.dockerignore
```

These support reproducible local and hosted deployment workflows.

## Docker

Build:

```bash
docker build -t fused .
```

Run:

```bash
docker run --rm -p 8501:8501 fused
```

Then open:

```text
http://localhost:8501
```

---

# Data and Secrets

Do not commit:

* API keys
* `.env` files
* local SQLite databases
* local virtual environments
* generated retrieval indexes
* benchmark caches
* large generated datasets
* the 100M-token corpus
* Streamlit secrets

Use environment variables for credentials such as:

```text
GEMINI_API_KEY
```

The repository `.gitignore` is configured to exclude common runtime and generated artifacts.

---

# Limitations

FUSED is intentionally experimental.

Important limitations:

1. The fallback backend is not a general-purpose language model.
2. Semantic retrieval requires optional dependencies and a suitable embedding model.
3. Large-data experiments are not the same as real-world knowledge evaluation.
4. Synthetic data can test system behavior but does not establish real-world utility.
5. The current retrieval architecture is research-scale and is not presented as distributed production infrastructure.
6. Live LLM performance depends on the selected provider/model and is not established by offline tests.
7. The project contains experimental research components whose usefulness should be evaluated empirically rather than assumed from their names or architecture.
8. Free-form full-scale benchmarks can require significantly more time and hardware than the default regression suite.

---

# Reproducibility

FUSED separates:

```text
Source code
    +
small benchmark/evaluation data
    +
generated stress data
```

The large synthetic corpus is distributed separately so Git history remains manageable.

Synthetic datasets include provenance information such as:

* source seed
* generator version
* split
* generation method
* generator seed
* record hash

This makes large-data experiments easier to audit and reproduce.

---

# Research Timebox History

The current system was developed incrementally using Scrum-style timeboxes:

```text
TB01  Test / CI baseline
TB02  Evaluation contract
TB03  Language-model interface
TB04  Gemini backend
TB05  Semantic RAG
TB06  Supervised learning
TB07  FractalBrain routing
TB08  Large-data ingestion
TB09  Synthetic-data quality / provenance
TB10  100M-token stress corpus
TB11  Persistent retrieval index
TB12  Beginner browser UI
TB13  UI/UX refinement + 100M-token controls
```

The timebox history is retained as development evidence and documentation.

---

# Where to Start Reading the Code

For the main pipeline:

```text
README.md
    ↓
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
    ↓
language_model.py
    ↓
gemini_backend.py
```

For the beginner experience:

```text
docs/BEGINNER_TUTORIAL.md
```

For the browser UI:

```text
app/streamlit_app.py
```

For large-data work:

```text
large_data.py
synthetic_data.py
evaluation.py
```

---

# License

FUSED is released under the MIT License.

See:

```text
LICENSE
```

for the full license text.

Third-party or vendored components may have separate attribution or license requirements; review the corresponding documentation before redistribution.

---

# Summary

FUSED is best understood as an **experimental hybrid AI framework** rather than a single AI model.

Its main research idea is to combine:

```text
Memory
+
Semantic RAG
+
Planning
+
FractalBrain control
+
LLM generation
+
Feedback learning
+
Optimization
+
Measurement
```

into one reproducible pipeline that can be inspected, benchmarked, taught, and deployed through a beginner-friendly interface.
