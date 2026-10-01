"""Regenerates a win/tie/loss tally for the 4 realistic-synthetic-data XGBoost
problems from the run JSONs in runs/. Run from anywhere:
    python aggregate_realistic_synth_summary.py
"""
import json
import glob
import re
from pathlib import Path
from collections import defaultdict

RUNS_DIR = Path(__file__).with_name("runs")
PROBLEMS = ["synthetic_xgboost_gaussian", "synthetic_xgboost_gmm", "synthetic_xgboost_kde", "synthetic_xgboost_bootstrap"]

data = defaultdict(dict)
seeds = set()

for f in glob.glob(str(RUNS_DIR / "*.json")):
    d = json.load(open(f))
    if d.get("problem") not in PROBLEMS or d.get("trials") != 30:
        continue  # skip any smaller sanity-check runs mixed into runs/
    # the summary doesn't carry a top-level "seed" field; recover it from the
    # original output directory name embedded in summary_path (.../seed<N>/runs/...)
    match = re.search(r"seed(\d+)", d.get("summary_path", "") or d.get("csv_path", ""))
    if not match:
        continue
    seed = int(match.group(1))
    seeds.add(seed)
    data[(d["problem"], seed)][d["optimizer"]] = d["best_value"]

seeds = sorted(seeds)
print(f"{'problem':<28}{'seed':<6}{'ours':>10}{'optuna':>10}{'winner':>10}{'margin_%':>10}")
tally = defaultdict(int)
for p in PROBLEMS:
    for seed in seeds:
        row = data[(p, seed)]
        if "ours" not in row or "optuna" not in row:
            continue
        o, q = row["ours"], row["optuna"]  # rmse, minimize -> lower is better
        if abs(o - q) < 1e-9:
            winner = "tie"
        elif o < q:
            winner = "ours"
        else:
            winner = "optuna"
        tally[winner] += 1
        margin = (o - q) / q * 100
        print(f"{p:<28}{seed:<6}{o:>10.3f}{q:>10.3f}{winner:>10}{margin:>+9.1f}%")

print()
print(f"tally across {sum(tally.values())} (problem,seed) pairs:", dict(tally))
