# Optimization

FUSED contains two related optimization layers.

## Adaptive optimizer

`adaptive_optimizer.py` provides a dependency-free search implementation used by `pipeline_optimizer.py`.

The pipeline can search its own configuration values, for example:

- retrieval threshold
- planner sizing
- knowledge-graph settings
- other registered parameters

Run:

```bash
python hybrid_cli.py --mode tune
```

The number of trials can be changed with the CLI's trial controls.

## MetaTune

`metatune/` contains a higher-level optimization API and benchmark framework.

Example public API:

```python
from metatune import Study, Float, Int, Categorical
```

Optional numerical acceleration is available through:

```bash
pip install -e ".[fast]"
```

Benchmark dependencies are installed with:

```bash
pip install -e ".[benchmarks]"
```

## Interpretation

The optimizer is a research component. Benchmark results are workload- and seed-dependent. Avoid presenting a single benchmark run as a universal performance claim.
