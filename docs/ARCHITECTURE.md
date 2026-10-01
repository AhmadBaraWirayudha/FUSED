# Architecture

FUSED is a modular pipeline. The most important distinction is that language generation and the control/memory layers are separate.

```text
                         FUSED
                           │
        ┌──────────────────┼──────────────────┐
        ↓                  ↓                  ↓
      Memory            Control           Language
        │                  │                  │
   SQLite memory      FractalBrain        Backend API
   Semantic RAG       Router              Gemini
   Persistent index   Planner             Fallback
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ↓
                    Final response
                           ↓
                 Feedback / learning
```

## Main execution flow

1. Normalize the input.
2. Retrieve relevant memory.
3. Decompose the task.
4. Build a plan.
5. Run the FractalBrain confidence calculation.
6. Select a routing policy.
7. Generate through the configured language backend.
8. Produce a trace/evidence record.
9. Store explicit supervised corrections when supplied.

## Important modules

| Module | Responsibility |
|---|---|
| `ai_pipeline.py` | Unified orchestration |
| `engine.py` | Closed-loop execution |
| `memory.py` | Database-backed memory and retrieval |
| `decomposer.py` | Subtask extraction |
| `planner.py` | Planning |
| `decoder.py` | Generation path |
| `language_model.py` | Backend interface |
| `gemini_backend.py` | Gemini provider implementation |
| `fractal_router.py` | Policy selection from FractalBrain confidence |
| `fractal_brain/` | Custom control/learning subsystem |
| `lkg.py` | Living Knowledge Graph |
| `large_data.py` | Batch supervised ingestion |
| `evaluation.py` | Fixed evaluation |
| `learning_evaluation.py` | Learning evaluation |

## FractalBrain boundary

FractalBrain is a custom research subsystem. Its confidence is a control signal; it is not treated as a calibrated probability of correctness.

## Retrieval boundary

The normal retrieval system is local and persistent. Optional Sentence Transformer embeddings and optional FAISS change the retrieval implementation, not the rest of the pipeline.

## Compatibility layers

`hybrid_engine.py` preserves the unified entry path while root closed-loop modules remain available for compatibility. `ocle_clean_build/` is retained for package-style imports.
