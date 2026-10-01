# benchmarks package

Windows-native benchmark suite for validating optimizer overhead on CPU-only machine learning problems.

## Goals
- Compare `ours`, `optuna`, and `raytune` under identical search spaces and trial budgets.
- Measure wall-clock time, objective time, and optimizer overhead.
- Keep the suite runnable on Windows 10 without WSL or Docker.
- Produce CSV logs, plots, and a Markdown report.

## Main commands

```bash
python -m benchmarks.cli list-problems
python -m benchmarks.cli run --problem svm --optimizers ours optuna raytune --trials 40 --backend serial
python -m benchmarks.cli run-ray --problem mnist --trials 30 --workers 8
python -m benchmarks.cli profile --problem svm --optimizer ours --calls 10000
python -m benchmarks.cli report
python -m benchmarks.cli suite --trials 10 --workers 4
```

## Problems
- Digits SVM
- Breast Cancer LightGBM
- Diabetes XGBoost
- MNIST-like MLP with OpenML-first loading and offline fallback
- Imbalanced classification pipeline

## Notes
- Optional dependencies are used when installed.
- Missing optional packages fall back to local scikit-learn implementations.
- The suite caches objectives per problem and seed so process and Ray runs do not rebuild the objective for every trial.
- `BenchmarkRunner` accepts either an optimizer name or a prebuilt adapter instance.
