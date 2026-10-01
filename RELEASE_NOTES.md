# 0.4.0 Release Notes

TB12 converts FUSED into a browser-first research workspace for non-programmers.

## Start

- Windows: double-click `START_FUSED_UI.bat`.
- The launcher creates `.venv` when needed and installs the FUSED core, Streamlit UI, and pytest.
- Open `app/streamlit_app.py` directly with Streamlit for manual development.

## Benchmark

The UI exposes fixed evaluation, supervised-learning evaluation, and bounded large-data ingestion tests. Reports are written under `var/benchmark/` and can be downloaded from the browser.

## Deployment

- Streamlit Community Cloud entrypoint: `app/streamlit_app.py`
- Dependency file: `app/requirements.txt`
- Docker entrypoint is included via `Dockerfile`.
- Keep the 100M-token synthetic corpus outside normal Git history.

## Scope

TB12 does not claim production-scale distributed retrieval, real-world synthetic-data utility, or live Gemini quality without a real provider credential.

# TB11 Release Notes

TB11 adds persistent retrieval indexing, index inspection/rebuild commands, and a layperson-friendly Windows launcher/tutorial.

The persistent index uses a memory-mapped NumPy float32 backend by default when FAISS is not available, and uses FAISS when available.

A 20,000-record prefix of the TB10 corpus was successfully ingested and retrieved after process restart. The full 179,619-record corpus was not claimed as a completed ingest benchmark because the pure-Python embedding path exceeded the execution window.

# Release Notes

## 0.3.1

Fixed, in response to an external packaging/release audit (see CHANGELOG.md for the
full investigation of each finding):
- **config.yaml and data/bootstrap_dataset.jsonl are now actually bundled into the
  installable wheel**, via a new `runtime_data` package. Confirmed broken with a real
  `pip install` into a clean venv before fixing (the earlier sdist-only check had
  missed this), and confirmed fixed the same way. `OpenClosedLoopEngine.
  from_default_config()`/`UnifiedAIPipeline()` now fall back to the packaged copy when
  the repo-root sibling file isn't installed.
- **The outer project's version is now single-sourced** from new `_version.py`
  (`pyproject.toml` reads it dynamically) instead of being a separately-maintained
  literal. `fractal_brain/version.py` stays at `0.1.0` deliberately -- it tracks that
  sub-package's own, unchanged history -- with both files now commented to make the
  relationship explicit rather than looking like undocumented drift.
- New `hybrid_cli.py --version` flag, for parity with `python -m fractal_brain
  --version`.
- New `tests/test_packaging_install.py` -- builds a real wheel, installs it into a
  fresh venv, and runs it from outside the source tree with zero explicit config, so
  this class of bug can't ship silently again.
- `tests/test_smoke_suite_check_count_matches_documentation`,
  `tests/test_runtime_data_mirrors_the_real_config_and_dataset`,
  `tests/test_pyproject_version_is_dynamic_from_version_py` -- guard rails for the
  three fixes above.

Investigated and not changed: the "137 checks" figure documented in several places was
found to be accurate (not stale) on closer inspection -- see CHANGELOG.md. The
root-level-modules-vs-`fractal_brain/`-package split was confirmed as a real,
correctly-identified architectural trade-off and documented in
`docs/UNIFIED_PIPELINE.md`'s "Known limitations" rather than restructured.

## 0.3.0

Added:
- `adaptive_optimizer.py` -- a pure-stdlib hyperparameter optimization engine (21+
  search strategies, Bayesian belief tracking, GP-based acquisition, constraints,
  multi-objective scoring), merged in from a standalone module and wired to the real
  ecosystem via new `pipeline_optimizer.py`: adapters bridge its knowledge-graph and
  memory dependency-injection points to the real `lkg.py` graph (an `optimizer_state`
  entity tracked the same way `session_intent` is, plus a real "still improving" fact
  per completed trial) and `fractal_brain.storage.Storage` (elite/failed configuration
  persistence, in a dedicated `data/optimizer.db` -- its schema collides with
  `VectorMemoryStore`'s if pointed at the same file as `data/engine.db`, see
  `CHANGELOG.md`). A real, scoped parameter registry (8 parameters, every one mapping
  to an actual `config.yaml` key) replaces the module's own aspirational defaults, and
  a real objective function scores a candidate configuration by actually building and
  running the pipeline against benchmark prompts. See `docs/ADAPTIVE_OPTIMIZER.md` for
  the full design and the two bugs found and fixed while building the objective
  function (it was initially insensitive to every registered parameter; a path
  resolution bug independently left every trial's memory corpus empty).
- `--mode tune` in `hybrid_cli.py` (`python hybrid_cli.py --mode tune --trials 20`).
- `engine.dump_simple_yaml`, the inverse of the existing `parse_simple_yaml`, needed to
  write a trial's overridden configuration to a real temp file.
- `paths.optimizer_db` in `config.yaml`.
- `tests/test_pipeline_optimizer.py`.

## 0.2.0

Added:
- `lkg.py` -- a Living Knowledge Graph (Beta-Bernoulli fact confidence,
  per-source reliability, particle-filtered entity-state tracking), merged in
  from a standalone module and wired into `OpenClosedLoopEngine`/
  `UnifiedAIPipeline`: a `session_intent` entity tracks each turn's task
  intent, and each retrieved document's match to that intent is a fact
  reinforced or penalized once real feedback closes the loop (confidence is
  purely feedback-driven -- a document with no feedback yet reads as
  neutral, not inflated by retrieval alone), attributed to the document's
  origin (bootstrap data, teacher feedback, or a prior successful
  interaction). Surfaced in the engine/pipeline result (`knowledge_graph`),
  the reflection (topic-shift and low-confidence-source risk notes), the
  pipeline trace, and generation itself: `SharedMoEBackbone` now appends a
  caution line when a retrieved document it's about to surface verbatim has
  a poor knowledge-graph track record for this intent, even if its cosine
  similarity was strong. See `docs/UNIFIED_PIPELINE.md` and `CHANGELOG.md`
  for the full design and the functional gaps found and fixed along the way
  (entity-state observation had no public entry point; an earlier version
  of the confidence design had a per-document floor tied to retrieval score;
  the caution line could go stale if echoed back from stored text).
- `knowledge_graph:` section in `config.yaml`, including a
  `low_confidence_threshold` shared by the reflection and the generation
  caveat above.
- `tests/test_lkg.py` (the module's own test suite, plus new coverage for
  the methods added during integration), `tests/test_knowledge_graph_integration.py`,
  and knowledge-graph regression tests in `tests/test_regressions.py`.

## 0.1.0

Added:
- `pyproject.toml`
- `requirements.txt`
- `LICENSE`
- CI workflow
- API reference
- architecture diagram
- training guide
- package version export

These files cover the main missing project-level scaffolding identified in the review.


## TB07 — FractalBrain Routing

FUSED now converts the pre-generation FractalBrain confidence estimate into an explicit downstream generation policy: `direct_grounded`, `standard`, or `cautious_reasoning`. Gemini consumes route-specific thinking/budget settings, and the fallback backend changes its output policy accordingly.

## TB13

The Streamlit UI is redesigned for beginner use. The 100M-token stress corpus now has a dedicated UI section for detection, auditing, generation, and ingestion.
