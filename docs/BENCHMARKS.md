# Benchmarks

FUSED has two different benchmark families. They should not be mixed.

## A. FUSED pipeline benchmarks

These measure the AI pipeline itself:

- answer correctness
- retrieval Hit@1 / Hit@K
- MRR
- latency
- learning before/after correction
- data ingestion throughput

Use:

```bash
python hybrid_cli.py --mode evaluate
python hybrid_cli.py --mode evaluate-learning
```

See [`DATA_AND_BENCHMARKS.md`](DATA_AND_BENCHMARKS.md).

## B. MetaTune optimization benchmarks

`metatune/benchmarks/` compares optimization behavior on benchmark problems. The project has historical comparisons against Optuna and synthetic benchmark workloads.

Run the benchmark CLI after installing the benchmark extra:

```bash
pip install -e ".[benchmarks]"
```

Then inspect:

```bash
python -m metatune.benchmarks --help
```

## How to interpret results

Do not infer a universal winner from one run. Optimization results depend on:

- problem
- search budget
- seed
- dependency versions
- hardware
- objective definition

The repository's historical benchmark artifacts have therefore been removed from the curated source tree; reproducible benchmark outputs should be generated locally and retained as release evidence when needed.
