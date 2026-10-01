"""Single source of truth for the outer "hybrid-fused-brain" project's own
version -- pyproject.toml reads it dynamically (`[tool.setuptools.dynamic]`)
rather than duplicating the number as a separate literal, and hybrid_cli.py
--version reports it at runtime the same way `python -m fractal_brain
--version` reports fractal_brain/version.py's.

This is intentionally a *different* number from fractal_brain.__version__,
not a drifted duplicate of it: this tracks the outer integration project
(engine.py/ai_pipeline.py/lkg.py/adaptive_optimizer.py/.... -- see
CHANGELOG.md for what's shipped at each version), while fractal_brain's own
version tracks just that sub-package, which has its own independent history
and hasn't changed while this one has. tests/test_package_metadata.py checks
both stay internally consistent (this file matches what pyproject.toml's
build actually resolves to; fractal_brain's matches its own).
"""
__version__ = "0.4.0"
