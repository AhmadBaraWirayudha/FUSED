"""Builds a real wheel, installs it into a fresh venv, and runs the
installed package from outside the source tree with zero explicit config
path -- i.e., actually reproduces what a `pip install` user would do,
rather than checking file listings.

This exists because that specific gap is exactly how a real bug shipped
undetected: sdist file-listing checks (`python -m build --sdist` + grep the
tarball) passed, but the built *wheel* silently omitted config.yaml and
data/bootstrap_dataset.jsonl, and a real `pip install` + run from a clean
venv raised FileNotFoundError immediately. See CHANGELOG.md, "config.yaml
and the bootstrap dataset bundled into the distributable" for the full
story and the fix (runtime_data package + OpenClosedLoopEngine.
from_packaged_default_config() fallback).

Slower than the rest of the suite (builds a wheel and creates a venv --
budget ~20-40s) and skipped if `python -m venv`/`pip` aren't usable in this
environment, since that's an environment limitation, not a regression.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _venv_python(venv_dir: Path) -> Path:
    candidate = venv_dir / 'bin' / 'python3'
    return candidate if candidate.exists() else venv_dir / 'Scripts' / 'python.exe'


@pytest.mark.slow
def test_installed_wheel_works_with_zero_explicit_config(tmp_path) -> None:
    dist_dir = tmp_path / 'dist'
    venv_dir = tmp_path / 'venv'

    build = subprocess.run(
        [sys.executable, '-m', 'build', '--wheel', '--outdir', str(dist_dir), str(REPO_ROOT)],
        capture_output=True, text=True,
    )
    if build.returncode != 0 and 'No module named build' in (build.stderr or ''):
        pytest.skip('the "build" package is not installed in this environment')
    assert build.returncode == 0, build.stderr
    # python -m build stages a build/ dir and a *.egg-info/ dir inside
    # REPO_ROOT (the source it builds from), not inside tmp_path -- clean
    # those up here rather than leaving them in the actual source tree.
    shutil.rmtree(REPO_ROOT / 'hybrid_fused_brain.egg-info', ignore_errors=True)
    shutil.rmtree(REPO_ROOT / 'build', ignore_errors=True)

    wheels = list(dist_dir.glob('*.whl'))
    assert len(wheels) == 1, f'expected exactly one wheel, found {wheels}'

    venv.create(venv_dir, with_pip=True)
    python = _venv_python(venv_dir)

    install = subprocess.run(
        [str(python), '-m', 'pip', 'install', '-q', str(wheels[0])],
        capture_output=True, text=True,
    )
    assert install.returncode == 0, install.stderr

    # Run from tmp_path itself (well outside the source tree, and not the
    # repo root), with zero explicit config path -- exactly what a
    # pip-installed user with no source checkout would do.
    probe_script = tmp_path / 'probe.py'
    probe_script.write_text(
        "from engine import OpenClosedLoopEngine\n"
        "from ai_pipeline import UnifiedAIPipeline\n"
        "e = OpenClosedLoopEngine.from_default_config()\n"
        "e.initialize()\n"
        "assert len(e.memory.documents) > 0, 'bootstrap dataset did not load'\n"
        "p = UnifiedAIPipeline()\n"
        "p.initialize()\n"
        "r = p.run('Solve the integral of 2x from 0 to 4.')\n"
        "assert r.final_output\n"
        "print('OK')\n",
        encoding='utf-8',
    )
    run = subprocess.run(
        [str(python), str(probe_script)],
        capture_output=True, text=True, cwd=str(tmp_path),
    )
    assert run.returncode == 0, f'stdout:\n{run.stdout}\nstderr:\n{run.stderr}'
    assert run.stdout.strip() == 'OK'
