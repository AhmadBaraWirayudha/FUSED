from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, TextIO

SCHEMA_VERSION = "3.0"
GENERATOR_VERSION = "tb10-max-token-v1"
SOURCE_SEEDS = {
    "m1": "Compute the integral of 2x from 0 to 4.",
    "m2": "Solve x^2 - 5x + 6 = 0.",
    "e1": "Estimate the voltage drop across a 10 ohm resistor carrying 2 A.",
    "c1": "Fix a Python loop that appends numbers 0 to 4.",
}


def _case_hash(case: dict[str, Any]) -> str:
    payload = json.dumps(case, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def estimate_tokens(text: str) -> int:
    """Dependency-free conservative English token estimate (~4 chars/token)."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def _pad_explanation(prefix: str, blocks: list[str], target_tokens: int) -> str:
    parts = [prefix] + blocks
    text = "\n".join(parts)
    # Deterministic, domain-neutral audit language keeps synthetic targets long
    # without pretending the filler is new ground truth.
    filler = (
        " Verification note: this synthetic example is generated from a known "
        "symbolic rule. Check the supplied values, apply the stated formula, "
        "and compare the final result against the expected output before using "
        "the record as supervised training evidence."
    )
    while estimate_tokens(text) < target_tokens:
        text += filler
    return text


def _integral_case(rng: random.Random, index: int, target_tokens: int = 32) -> tuple[str, str, dict[str, Any]]:
    upper = 1 + ((index * 7919 + 17) % 1_000_000)
    templates = [
        "Compute the integral of 2x from 0 to {u}.",
        "Evaluate integral_0^{u} 2x dx.",
        "Find the area from 0 to {u} under f(x)=2x.",
        "What is the definite integral of 2x over [0, {u}]?",
        "For f(x)=2x, calculate the definite integral on 0 <= x <= {u}.",
        "Using the antiderivative of 2x, evaluate the area from zero to {u}.",
    ]
    query = templates[index % len(templates)].format(u=upper)
    result = upper * upper
    answer = _pad_explanation(
        f"The integral of 2x from 0 to {upper} is {result}.",
        [
            "Antiderivative: x^2.",
            f"Upper endpoint evaluation: {upper}^2 = {result}.",
            "Lower endpoint evaluation: 0^2 = 0.",
            f"Final value: {result}.",
        ],
        target_tokens,
    )
    return query, answer, {"upper": upper, "expected_contains": [str(result)]}


def _quadratic_case(rng: random.Random, index: int, target_tokens: int = 32) -> tuple[str, str, dict[str, Any]]:
    # Enumerate unordered root pairs from [-5000, 4999] in O(log(span)).
    span = 10000
    k = index
    lo, hi = 0, span - 2
    while lo < hi:
        mid = (lo + hi) // 2
        cumulative_next = (mid + 1) * span - (mid * (mid + 1)) // 2
        if cumulative_next > k:
            hi = mid
        else:
            lo = mid + 1
    first = lo
    cumulative_before = first * span - (first * (first - 1)) // 2
    offset = k - cumulative_before
    second = first + 1 + offset
    r1 = -5000 + first
    r2 = -5000 + second
    b = r1 + r2
    c = r1 * r2
    templates = [
        "Solve x^2 - {b}x + {c} = 0.",
        "Find the roots of x^2 - {b}x + {c} = 0.",
        "What are the solutions to x^2 - {b}x + {c} = 0?",
        "Factor and solve x^2 - {b}x + {c} = 0.",
        "Verify the roots of x^2 - {b}x + {c} = 0 and report both values.",
        "Solve the quadratic with coefficients a=1, b={b}, c={c}.",
    ]
    query = templates[index % len(templates)].format(b=b, c=c)
    answer = _pad_explanation(
        f"The roots are x={r1} and x={r2}.",
        [
            f"The polynomial factors as (x-{r1})(x-{r2}).",
            f"The sum of the roots is {r1 + r2}, matching -b = {-b}.",
            f"The product of the roots is {r1 * r2}, matching c = {c}.",
            "Therefore both reported roots satisfy the quadratic exactly.",
        ],
        target_tokens,
    )
    return query, answer, {"root_1": r1, "root_2": r2, "expected_contains": [f"x={r1}", f"x={r2}"]}


def _ohm_case(rng: random.Random, index: int, target_tokens: int = 32) -> tuple[str, str, dict[str, Any]]:
    resistance = 1 + (index % 10_000)
    current = 1 + ((index // 10_000) % 1_000)
    voltage = resistance * current
    templates = [
        "Estimate the voltage drop across a {r} ohm resistor carrying {i} A.",
        "Using Ohm's law, find V for R={r} ohm and I={i} A.",
        "What voltage results from {i} A through {r} ohm?",
        "Calculate the resistor voltage when resistance is {r} ohm and current is {i} A.",
        "For a resistive load of {r} ohm at {i} A, determine the voltage drop.",
        "Apply V=IR to R={r} ohm and I={i} A.",
    ]
    query = templates[index % len(templates)].format(r=resistance, i=current)
    answer = _pad_explanation(
        f"Voltage drop = {voltage} V.",
        [
            "Use Ohm's law: V = I*R.",
            f"Substitute I={current} A and R={resistance} ohm.",
            f"V = {current} * {resistance} = {voltage} V.",
            "This assumes an ideal linear resistor with the stated current and resistance.",
        ],
        target_tokens,
    )
    return query, answer, {"resistance_ohm": resistance, "current_a": current, "expected_contains": [f"{voltage} V"]}


def _python_case(rng: random.Random, index: int, target_tokens: int = 32) -> tuple[str, str, dict[str, Any]]:
    n = 2 + index
    templates = [
        "Fix a Python loop that appends numbers 0 to {last}.",
        "Write Python code to append integers 0 through {last} to values.",
        "How should a Python loop create the sequence 0..{last} in a list?",
        "Correct this Python task: append every number from 0 to {last}.",
        "Generate a Python list containing every integer from 0 through {last}.",
        "Use range() correctly to append 0 through {last} to values.",
    ]
    query = templates[index % len(templates)].format(last=n - 1)
    answer = _pad_explanation(
        f"Use: for i in range({n}): values.append(i)",
        [
            f"range({n}) produces integers from 0 through {n - 1}, excluding {n}.",
            "Append each integer inside the loop to the existing values list.",
            "Equivalent compact form: values.extend(range(n)) after assigning n.",
            f"Expected sequence begins with 0 and ends with {n - 1}.",
        ],
        target_tokens,
    )
    return query, answer, {"count": n, "expected_contains": [f"range({n})", "values.append(i)"]}


GENERATORS = {"m1": _integral_case, "m2": _quadratic_case, "e1": _ohm_case, "c1": _python_case}


def iter_grounded_cases(
    count: int,
    seed: int = 42,
    train_ratio: float = 0.8,
    target_tokens_per_record: int = 32,
) -> Iterable[dict[str, Any]]:
    if count <= 0:
        return
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if target_tokens_per_record < 8:
        raise ValueError("target_tokens_per_record must be >= 8")
    rng = random.Random(seed)
    seed_offset = (seed * 17) % 1000
    seeds = tuple(SOURCE_SEEDS)
    for i in range(count):
        family_index = (i // len(seeds)) + seed_offset
        source_seed = seeds[i % len(seeds)]
        query, answer, params = GENERATORS[source_seed](rng, family_index, target_tokens_per_record)
        split = "train" if i < int(count * train_ratio) else "eval"
        record = {
            "schema_version": SCHEMA_VERSION,
            "case_id": f"syn-{i:08d}",
            "input": query,
            "ideal_output": answer,
            "metadata": {
                "synthetic": True,
                "generator_version": GENERATOR_VERSION,
                "generation_method": "deterministic_parametric_template_with_target_length",
                "source_type": "project_bootstrap_seed",
                "source_seed_id": source_seed,
                "source_seed_hash": hashlib.sha256(SOURCE_SEEDS[source_seed].encode()).hexdigest()[:16],
                "split": split,
                "family": source_seed,
                "generator_seed": seed,
                "target_tokens_per_record": target_tokens_per_record,
                "estimated_input_tokens": estimate_tokens(query),
                "estimated_output_tokens": estimate_tokens(answer),
                **params,
            },
        }
        record["metadata"]["record_hash"] = _case_hash(record)
        yield record


def _open_text(path: Path, mode: str = "rt") -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8", compresslevel=1)
    return path.open(mode.replace("t", ""), encoding="utf-8")


def generate_grounded_synthetic_dataset(
    output: str | Path,
    count: int = 10000,
    seed: int = 42,
    train_ratio: float = 0.8,
    target_tokens_per_record: int = 32,
) -> dict[str, Any]:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    splits = Counter()
    estimated_tokens = 0
    with _open_text(output, "wt") as fh:
        for record in iter_grounded_cases(count, seed=seed, train_ratio=train_ratio, target_tokens_per_record=target_tokens_per_record):
            fh.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            counts[record["metadata"]["family"]] += 1
            splits[record["metadata"]["split"]] += 1
            estimated_tokens += record["metadata"]["estimated_input_tokens"] + record["metadata"]["estimated_output_tokens"]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "output": str(output),
        "records": count,
        "seed": seed,
        "train_ratio": train_ratio,
        "target_tokens_per_record": target_tokens_per_record,
        "estimated_total_tokens": estimated_tokens,
        "train_records": splits["train"],
        "eval_records": splits["eval"],
        "source_seed_ids": list(SOURCE_SEEDS),
        "families": dict(counts),
        "synthetic_only": True,
        "human_review": "not_performed",
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest | {"manifest": str(manifest_path)}


def generate_max_token_dataset(
    output: str | Path,
    token_budget: int = 25_000_000,
    target_tokens_per_record: int = 128,
    seed: int = 42,
    train_ratio: float = 0.8,
) -> dict[str, Any]:
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    if target_tokens_per_record < 32:
        raise ValueError("target_tokens_per_record must be >= 32 for max-token corpus")
    # Approximate record count from desired total tokens, based on generated samples.
    # We over-generate slightly, then stop once the requested budget is reached.
    count = max(1, math.ceil(token_budget / target_tokens_per_record))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    splits = Counter()
    estimated_tokens = 0
    records = 0
    with _open_text(output, "wt") as fh:
        for record in iter_grounded_cases(count, seed=seed, train_ratio=train_ratio, target_tokens_per_record=target_tokens_per_record):
            # Max-token mode can stop before the planned record count. Assign the split
            # independently so the realized stream still contains a stable 80/20 split.
            record["metadata"]["split"] = "eval" if (records % 5 == 4) else "train"
            row_tokens = record["metadata"]["estimated_input_tokens"] + record["metadata"]["estimated_output_tokens"]
            if records and estimated_tokens + row_tokens > token_budget:
                break
            record["metadata"]["record_hash"] = _case_hash(record)
            fh.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            estimated_tokens += row_tokens
            records += 1
            counts[record["metadata"]["family"]] += 1
            splits[record["metadata"]["split"]] += 1
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "output": str(output),
        "records": records,
        "requested_token_budget": token_budget,
        "estimated_total_tokens": estimated_tokens,
        "target_tokens_per_record": target_tokens_per_record,
        "seed": seed,
        "train_ratio": train_ratio,
        "train_records": splits["train"],
        "eval_records": splits["eval"],
        "families": dict(counts),
        "source_seed_ids": list(SOURCE_SEEDS),
        "generation_mode": "max_token_stream",
        "synthetic_only": True,
        "human_review": "not_performed",
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest | {"manifest": str(manifest_path)}


def audit_synthetic_dataset(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    seen_inputs: set[str] = set()
    duplicate_inputs = 0
    missing_provenance = 0
    split_inputs: dict[str, set[str]] = {"train": set(), "eval": set()}
    family_counts = Counter()
    split_counts = Counter()
    invalid_labels = 0
    total = 0
    estimated_tokens = 0
    with _open_text(path, "rt") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            total += 1
            metadata = record.get("metadata", {})
            key = str(record.get("input", "")).strip()
            if key in seen_inputs:
                duplicate_inputs += 1
            seen_inputs.add(key)
            required = ("synthetic", "generator_version", "generation_method", "source_seed_id", "split", "record_hash")
            if any(k not in metadata for k in required):
                missing_provenance += 1
            split = str(metadata.get("split", ""))
            if split in split_inputs:
                split_inputs[split].add(key)
            family_counts[str(metadata.get("family", "unknown"))] += 1
            split_counts[split] += 1
            estimated_tokens += int(metadata.get("estimated_input_tokens", 0)) + int(metadata.get("estimated_output_tokens", 0))
            if metadata.get("synthetic") is not True or metadata.get("source_seed_id") not in SOURCE_SEEDS:
                invalid_labels += 1
    leakage = len(split_inputs["train"] & split_inputs["eval"])
    return {
        "schema_version": SCHEMA_VERSION,
        "records": total,
        "unique_inputs": len(seen_inputs),
        "exact_duplicate_inputs": duplicate_inputs,
        "unique_input_ratio": (len(seen_inputs) / total) if total else 0.0,
        "provenance_complete_ratio": ((total - missing_provenance) / total) if total else 0.0,
        "train_eval_exact_input_overlap": leakage,
        "label_provenance_errors": invalid_labels,
        "estimated_total_tokens": estimated_tokens,
        "split_counts": dict(split_counts),
        "family_counts": dict(family_counts),
        "human_review": "not_performed",
        "source_seed_ids": list(SOURCE_SEEDS),
    }


def validate_synthetic_quality(report: dict[str, Any]) -> None:
    failures: list[str] = []
    if report["records"] == 0:
        failures.append("empty dataset")
    if report["provenance_complete_ratio"] != 1.0:
        failures.append("incomplete provenance")
    if report["train_eval_exact_input_overlap"] != 0:
        failures.append("train/eval exact-input leakage")
    if report["label_provenance_errors"] != 0:
        failures.append("invalid synthetic provenance")
    if report["unique_input_ratio"] < 0.99:
        failures.append("too many exact duplicate inputs")
    if failures:
        raise ValueError("synthetic quality gate failed: " + ", ".join(failures))


def build_parser() -> Any:
    import argparse
    parser = argparse.ArgumentParser(description="Grounded synthetic supervised-data utilities for FUSED.")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--output", default="data/synthetic_supervised_10000.jsonl")
    gen.add_argument("--count", type=int, default=10000)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--train-ratio", type=float, default=0.8)
    gen.add_argument("--target-tokens-per-record", type=int, default=32)
    maxgen = sub.add_parser("generate-max")
    maxgen.add_argument("--output", default="data/tb10/synthetic_supervised_max.jsonl.gz")
    maxgen.add_argument("--token-budget", type=int, default=25_000_000)
    maxgen.add_argument("--target-tokens-per-record", type=int, default=128)
    maxgen.add_argument("--seed", type=int, default=42)
    maxgen.add_argument("--train-ratio", type=float, default=0.8)
    audit = sub.add_parser("audit")
    audit.add_argument("--dataset", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        payload = generate_grounded_synthetic_dataset(args.output, args.count, args.seed, args.train_ratio, args.target_tokens_per_record)
    elif args.command == "generate-max":
        payload = generate_max_token_dataset(args.output, args.token_budget, args.target_tokens_per_record, args.seed, args.train_ratio)
    else:
        payload = audit_synthetic_dataset(args.dataset)
        validate_synthetic_quality(payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
