# Data and Benchmarks

This document covers the normal evaluation data, supervised data, the 100M-token stress corpus, and the limits of synthetic-data evidence.

## 1. Fixed evaluation

The versioned evaluation dataset lives under:

```text
data/evaluation/v1.jsonl
```

Each record contains:

- `schema_version`
- `case_id`
- `input`
- `expected`
- `metadata`

The deterministic evaluator reports:

- answer correctness
- retrieval Hit@1 / Hit@K
- MRR
- mean / median / p95 / maximum warm latency

Run:

```bash
python hybrid_cli.py --mode evaluate
```

The default evaluator isolates cases with fresh state so feedback from one case cannot contaminate another.

## 2. Learning evaluation

Learning evaluation compares a baseline answer against a post-teaching answer.

```bash
python hybrid_cli.py --mode evaluate-learning
```

The learning path stores the correction together with the original question so that later retrieval can find the lesson.

## 3. Synthetic supervised data

Synthetic supervised records are generated from a small set of parameterized seed problems. The generator adds provenance and a train/eval split.

Example:

```bash
python hybrid_cli.py --mode generate-supervised \
  --dataset data/synthetic_supervised_10000.jsonl \
  --count 10000
```

Audit a corpus:

```bash
python synthetic_data.py audit \
  --dataset data/synthetic_supervised_10000.jsonl
```

Synthetic data is useful for stress testing and controlled experiments. It is not a substitute for real-world evaluation.

## 4. 100M-token stress corpus

The TB10 workflow generates by estimated token budget and writes directly to gzip-compressed JSONL.

```bash
python hybrid_cli.py --mode generate-max-supervised \
  --dataset data/tb10/synthetic_supervised_max.jsonl.gz \
  --token-budget 100000000 \
  --target-tokens-per-record 512 \
  --seed 20261001 \
  --train-ratio 0.8
```

The reference artifact contains approximately:

```text
179,619 records
99,999,890 estimated tokens
80% train / 20% eval
0 exact train/eval input overlap
100% provenance completeness
```

The corpus is a stress-test workload, not evidence of broad knowledge.

## 5. Ingestion

```bash
python hybrid_cli.py --mode ingest-supervised \
  --dataset data/tb10/synthetic_supervised_max.jsonl.gz \
  --db data/supervised.db \
  --batch-size 500
```

The ingestion path is streaming and batch-oriented. Generated data is kept outside normal Git history.

## 6. Persistent retrieval index

The database and its retrieval index are separate generated artifacts:

```text
data/
├── supervised.db
└── supervised.retrieval_index/
    ├── manifest.json
    ├── doc_ids.json
    └── vectors.float32.bin
```

The exact vector file can differ when FAISS is installed.

Inspect:

```bash
python hybrid_cli.py --mode index-status --config config.yaml --db data/supervised.db
```

Rebuild:

```bash
python hybrid_cli.py --mode rebuild-index --config config.yaml --db data/supervised.db
```

## 7. Scale-testing protocol

Do not jump directly to the whole 100M-token corpus. Use increasing workloads:

```text
5k → 20k → 50k → 100k+ records
```

Record at least:

- ingestion time
- records/sec
- memory use
- database size
- index size
- retrieval latency
- answer accuracy on held-out cases

The full 179,619-record ingestion remains a machine-dependent performance benchmark, not a repository release gate.

## 8. Synthetic-data evidence boundary

Synthetic data can prove that the software can ingest, search, and benchmark a large controlled workload. It does not prove real-world utility, truthfulness, or generalization. For those claims, add held-out real data and human-reviewed evaluation.
