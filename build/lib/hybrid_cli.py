from __future__ import annotations

import argparse
import json
from typing import Sequence

from engine import OpenClosedLoopEngine, parse_simple_yaml
from ai_pipeline import UnifiedAIPipeline
from fractal_brain import FractalBrain, set_seed
from _version import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='python hybrid_cli.py',
        description='Unified launcher for the closed-loop engine, fractal brain, and the single AI pipeline.',
    )
    parser.add_argument(
        '--version',
        action='version',
        version=__version__,
        help="Print this project's own version (see _version.py) and exit. "
             "Distinct from `python -m fractal_brain --version`, which reports "
             "just the fractal_brain sub-package's version -- see CHANGELOG.md.",
    )
    parser.add_argument(
        '--mode',
        choices=('pipeline', 'closed-loop', 'fractal', 'session', 'teach', 'tune', 'evaluate', 'evaluate-learning', 'ingest-supervised', 'generate-supervised'),
        default='pipeline',
        help='Select which subsystem to run.',
    )
    parser.add_argument(
        '--text',
        default='Solve the integral of 2x from 0 to 4.',
        help='Input text used by the demo execution path.',
    )
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Config path for the closed-loop engine.',
    )
    parser.add_argument(
        '--ideal-output',
        default='',
        help='Ground-truth output for teach mode.',
    )
    parser.add_argument(
        '--notes',
        default=None,
        help='Optional notes for teach mode.',
    )
    parser.add_argument(
        '--dataset',
        default='data/evaluation/v1.jsonl',
        help='Versioned JSONL evaluation dataset for evaluate mode.',
    )
    parser.add_argument(
        '--output',
        default='docs/baselines/v0.3.1_evaluation.json',
        help='Evaluation report output path.',
    )
    parser.add_argument(
        '--db',
        default='data/supervised.db',
        help='SQLite path for large-data supervised ingestion.',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=500,
        help='Batch size for large-data supervised ingestion.',
    )
    parser.add_argument(
        '--max-records',
        type=int,
        default=None,
        help='Optional cap for supervised ingestion.',
    )
    parser.add_argument(
        '--count',
        type=int,
        default=10000,
        help='Synthetic supervised records to generate.',
    )
    parser.add_argument(
        '--trials',
        type=int,
        default=10,
        help='Number of trials for tune mode. Budget roughly 5-10s/trial (see docs/ADAPTIVE_OPTIMIZER.md): each trial builds and runs a full pipeline instance.',
    )
    return parser


def run_closed_loop(text: str, config: str) -> dict:
    engine = OpenClosedLoopEngine(config)
    engine.initialize()
    return engine.run(text)


def run_fractal(text: str) -> dict:
    set_seed(7)
    brain = FractalBrain(vocab_size=32, d_model=8, num_experts=4)
    token_ids = [ord(ch) % 32 for ch in text[:16]] or [0]
    logits, loss = brain.step(token_ids, [0.0] * 32)
    return {
        'token_ids': token_ids,
        'logits_rows': logits.rows,
        'logits_cols': logits.cols,
        'loss': float(loss),
    }


def run_pipeline(text: str, config: str) -> dict:
    pipeline = UnifiedAIPipeline(config_path=config)
    pipeline.initialize()
    return pipeline.run(text).to_dict()


def run_session(text: str, config: str) -> dict:
    pipeline = UnifiedAIPipeline(config_path=config)
    pipeline.initialize()
    turns = [item.strip() for item in text.split('||') if item.strip()]
    return pipeline.run_session(turns)


def run_teach(text: str, config: str, ideal_output: str, notes: str | None) -> dict:
    pipeline = UnifiedAIPipeline(config_path=config)
    pipeline.initialize()
    return pipeline.teach_from_example(text, ideal_output or text, notes=notes)


def run_tune(config: str, trials: int) -> dict:
    from pathlib import Path
    from pipeline_optimizer import build_pipeline_optimizer

    config_path = Path(config)
    base_config = parse_simple_yaml(config_path.read_text(encoding='utf-8'))
    base_path = config_path.resolve().parent
    optimizer, objective = build_pipeline_optimizer(base_config, base_path)
    best = optimizer.optimize(objective, max_trials=trials, batch_size=1)
    return {
        'best_score': best.composite_score if best else None,
        'best_config': best.config if best else None,
        'summary': optimizer.get_summary_report(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.mode == 'closed-loop':
        payload = run_closed_loop(args.text, args.config)
    elif args.mode == 'fractal':
        payload = run_fractal(args.text)
    elif args.mode == 'session':
        payload = run_session(args.text, args.config)
    elif args.mode == 'teach':
        payload = run_teach(args.text, args.config, args.ideal_output, args.notes)
    elif args.mode == 'tune':
        payload = run_tune(args.config, args.trials)
    elif args.mode == 'evaluate':
        from evaluation import evaluate_dataset
        summary = evaluate_dataset(args.dataset, args.config)
        payload = summary.to_dict()
        from pathlib import Path
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    elif args.mode == 'ingest-supervised':
        from large_data import ingest_supervised_dataset
        payload = ingest_supervised_dataset(args.dataset, args.config, args.db, args.batch_size, args.max_records)
    elif args.mode == 'generate-supervised':
        from large_data import generate_synthetic_dataset
        payload = generate_synthetic_dataset(args.dataset, args.count)
    elif args.mode == 'evaluate-learning':
        from learning_evaluation import evaluate_learning
        summary = evaluate_learning(args.dataset, args.config)
        payload = summary.to_dict()
        from pathlib import Path
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    else:
        payload = run_pipeline(args.text, args.config)

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
