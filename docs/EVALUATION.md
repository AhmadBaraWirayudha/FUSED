# Evaluation

FUSED separates three questions:

1. Can the pipeline produce the required answer on a fixed test set?
2. Can retrieval find the intended memory?
3. Does explicit supervised teaching change later behavior?

## Fixed evaluation

Dataset:

```text
data/evaluation/v1.jsonl
```

Run:

```bash
python hybrid_cli.py --mode evaluate
```

The evaluator uses fresh isolated pipeline/database state per case and reports:

- answer correctness
- retrieval Hit@1 / Hit@K
- MRR
- mean / median / p95 / maximum warm latency

Answer correctness is a deterministic substring contract defined by each case's `expected.answer_contains` field. This is a regression metric, not a complete measure of answer quality.

## Learning evaluation

Dataset:

```text
data/evaluation/learning_v1.jsonl
```

Run:

```bash
python hybrid_cli.py --mode evaluate-learning
```

This measures baseline behavior before teaching and behavior after a supervised correction has been stored as question-linked memory.

## Real LLM runs

The offline fallback is deterministic enough for repeatable regression testing. External model responses are not treated as deterministic. Record provider/model metadata and keep live-model measurements separate from offline baselines.

## Scale evaluation

For large synthetic workloads, record:

- record count
- estimated tokens
- ingestion time
- records/sec
- database size
- index size
- retrieval latency
- held-out answer accuracy

The 100M-token corpus is a stress-test workload. Its size does not establish broad knowledge or real-world performance.

## Reproducibility rule

A benchmark result should always identify the dataset version, configuration, model/backend, seed where applicable, hardware/environment, and whether the run used real or synthetic data.
