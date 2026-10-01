from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

TARGETED = [
    'tests/test_tb12_ui.py',
    'tests/test_persistent_index.py',
    'tests/test_large_data.py',
    'tests/test_cli.py',
    'tests/test_evaluation.py',
    'tests/test_fractal_routing.py',
    'tests/test_language_model_interface.py',
    'tests/test_gemini_backend.py',
    'tests/test_learning_evaluation.py',
    'tests/test_synthetic_data.py',
]


def main() -> int:
    print('FUSED release verification')
    print('=' * 60)
    commands = [
        [sys.executable, '-m', 'pytest', '-q', *TARGETED],
        [sys.executable, 'tests/test_smoke.py'],
    ]
    for command in commands:
        print('\n>', ' '.join(command))
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode:
            print(f'FAILED with exit code {completed.returncode}')
            return completed.returncode
    print('\nALL RELEASE CHECKS PASSED')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
