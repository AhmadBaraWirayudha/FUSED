from fractal_brain import __version__, FractalBrain, set_seed


def test_version_and_imports() -> None:
    assert __version__ == "0.1.0"
    set_seed(1)
    brain = FractalBrain(vocab_size=16, d_model=8, num_experts=2)
    assert brain.vocab_size == 16


from pathlib import Path


def test_packaging_files_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "pyproject.toml").exists()
    assert (root / "LICENSE").exists()
    assert (root / "MANIFEST.in").exists()


def test_runtime_data_mirrors_the_real_config_and_dataset() -> None:
    """runtime_data/ is the packaged fallback OpenClosedLoopEngine.
    from_packaged_default_config() reads from when engine.py's sibling
    config.yaml isn't installed (a loose py_module can't carry adjacent
    data files via standard packaging -- see CHANGELOG.md, "config.yaml
    and the bootstrap dataset bundled into the distributable"). It's a
    second copy, not the same file, so nothing catches the two silently
    diverging except this test."""
    root = Path(__file__).resolve().parents[1]
    assert (root / "config.yaml").read_bytes() == (root / "runtime_data" / "config.yaml").read_bytes()
    assert (root / "data" / "bootstrap_dataset.jsonl").read_bytes() == (
        root / "runtime_data" / "data" / "bootstrap_dataset.jsonl"
    ).read_bytes()


def test_pyproject_version_is_dynamic_from_version_py() -> None:
    """pyproject.toml's version used to be a separate hardcoded literal
    that drifted from fractal_brain/version.py's (0.3.0 vs 0.1.0, flagged
    as confusing for downstream consumers -- see CHANGELOG.md). Fixed by
    single-sourcing it from _version.py via [tool.setuptools.dynamic]
    rather than by forcing the two to match, since they track different
    scopes (see _version.py's and fractal_brain/version.py's own
    docstrings/comments) -- this checks the dynamic-sourcing wiring itself
    stays intact without needing a tomllib/tomli dependency this
    zero-dependency project doesn't otherwise have.
    """
    from _version import __version__ as outer_version

    root = Path(__file__).resolve().parents[1]
    pyproject_text = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'dynamic = ["version"]' in pyproject_text
    assert 'version = {attr = "_version.__version__"}' in pyproject_text
    assert outer_version  # non-empty
    assert outer_version != __version__  # the two scopes are allowed, expected, to differ
