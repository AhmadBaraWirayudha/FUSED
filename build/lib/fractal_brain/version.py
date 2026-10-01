# fractal_brain's own version -- intentionally independent of the outer
# "hybrid-fused-brain" project's version (see /_version.py at the repo
# root, and CHANGELOG.md): this tracks only this sub-package's own history,
# which hasn't changed across several rounds of integration work on the
# OCLE/engine.py side. tests/test_package_metadata.py checks this file and
# _version.py each stay consistent with their own consumers -- it does not
# and should not assert the two match each other.
__version__ = "0.1.0"
