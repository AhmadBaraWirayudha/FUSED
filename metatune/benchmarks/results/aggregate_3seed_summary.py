"""Regenerates AGGREGATE_SUMMARY.txt from the run JSONs in runs/. Run from anywhere:
    python aggregate_3seed_summary.py
"""
import json
import glob
from pathlib import Path
from collections import defaultdict

RUNS_DIR = Path(__file__).with_name("runs")

direction = {
    "digits_svm": "maximize",
    "breast_cancer_lightgbm": "maximize",
    "diabetes_xgboost": "minimize",
    "mnist_like_mlp": "maximize",
    "imbalanced_pipeline": "maximize",
}

data = defaultdict(dict)
times = defaultdict(dict)
seeds = set()

for f in glob.glob(str(RUNS_DIR / "*.json")):
    d = json.load(open(f))
    seed = d.get("seed")
    if seed is None:
        continue
    seeds.add(seed)
    data[(d["problem"], seed)][d["optimizer"]] = d["best_value"]
    times[(d["problem"], seed)][d["optimizer"]] = d["mean_total_time_sec"] * d["trials"]

seeds = sorted(seeds)
problems = ["digits_svm", "breast_cancer_lightgbm", "diabetes_xgboost", "mnist_like_mlp", "imbalanced_pipeline"]

print(f"{'problem':<24}{'seed':<6}{'ours':>12}{'optuna':>12}{'winner':>10}")
tally = defaultdict(int)
for p in problems:
    for seed in seeds:
        row = data[(p, seed)]
        if "ours" not in row or "optuna" not in row:
            continue
        o, q = row["ours"], row["optuna"]
        d = direction[p]
        if abs(o - q) < 1e-9:
            winner = "tie"
        elif (d == "maximize" and o > q) or (d == "minimize" and o < q):
            winner = "ours"
        else:
            winner = "optuna"
        tally[winner] += 1
        print(f"{p:<24}{seed:<6}{o:>12.4f}{q:>12.4f}{winner:>10}")

print()
print(f"tally across {sum(tally.values())} (problem,seed) pairs:", dict(tally))
print()
print(f"{'problem':<24}{'ours avg wall(s)':>18}{'optuna avg wall(s)':>20}")
for p in problems:
    ot = [times[(p, s)]["ours"] for s in seeds if "ours" in times[(p, s)]]
    qt = [times[(p, s)]["optuna"] for s in seeds if "optuna" in times[(p, s)]]
    if ot and qt:
        print(f"{p:<24}{sum(ot)/len(ot):>18.2f}{sum(qt)/len(qt):>20.2f}")
