# API Reference

The public FractalBrain surface is re-exported by `fractal_brain.__init__`.

## Core

- `FractalBrain`
- `set_seed`
- `__version__`

## Utilities

- `Vector`
- `Matrix`
- `softmax`
- `softmax_rows`
- `kl_divergence`
- `sample_multinomial`

## Control/search

- `PIDController`
- `BootstrapGate`
- `FractalMarkovNode`
- `build_fractal_chain`
- `LassoTentacles`
- `Wormhole`
- `LogicFolder`
- `fold_states`
- `fuzzy_and`
- `fuzzy_or`
- `fuzzy_not`

## Learning/memory

- `TransformerExpert`
- `GatedMoE`
- `VectorStore`
- `StateRAGFusion`
- `BCMPlasticity`
- `JEPA`
- `Value`
- `distillation_loss`

## Representation/persistence

- `BPETokenizer`
- `normalize_text`
- `TextDataset`
- `DatasetView`
- `save_checkpoint`
- `load_checkpoint`
- `serialize_brain`
- `deserialize_brain`
- `Storage`

## Key behavior contracts

### `FractalBrain.step(token_ids, target_distribution)`

Advances the model state and returns `(logits, loss)`. With a target, trainable parameters are updated through the configured optimizer path.

### `FractalBrain.evaluate(token_ids, target_distribution)`

Read-only forward/evaluation path. It does not update weights or training state.

### `FractalBrain.train_batch(batch)`

Accepts `(token_ids, target_distribution)` pairs and applies an averaged trainable gradient update.

### `save_checkpoint(path)` / `load_checkpoint(path)`

Persist and restore model, optimizer, PID, storage, and RNG state for resume workflows.
