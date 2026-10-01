from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def test_smoke_suite_is_visible_to_pytest() -> None:
    """Bridge the project's standalone smoke suite into normal pytest output."""
    repo_root = Path(__file__).resolve().parents[1]
    smoke = repo_root / "tests" / "test_smoke.py"
    proc = subprocess.run(
        [sys.executable, str(smoke)],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )

    output = f"{proc.stdout}\n{proc.stderr}"
    match = re.search(r"(\d+) passed,\s*(\d+) failed", output)
    if match is None:
        raise AssertionError(
            "Standalone smoke suite did not emit its pass/fail summary.\n"
            f"returncode={proc.returncode}\n{output[-4000:]}"
        )

    passed, failed = (int(group) for group in match.groups())
    assert proc.returncode == 0, output[-4000:]
    assert failed == 0, output[-4000:]
    assert passed >= 156, f"Expected at least 156 smoke checks, got {passed}"
