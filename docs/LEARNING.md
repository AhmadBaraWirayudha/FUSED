# Supervised Learning

FUSED's current learning mechanism is memory-based supervised feedback.

## Workflow

```text
Question
   ↓
FUSED answer
   ↓
Human correction
   ↓
Question + corrected answer stored
   ↓
Future retrieval
   ↓
Corrected answer becomes available
```

Run the benchmark:

```bash
python hybrid_cli.py --mode evaluate-learning
```

The important design choice is storing the original question together with the correction. That makes the lesson retrievable when the same or a related question appears later.

## Evidence boundary

A small controlled learning benchmark can demonstrate that a correction is retrievable after teaching. It does not establish broad continual-learning capability, robust generalization, or model weight improvement across arbitrary domains.
