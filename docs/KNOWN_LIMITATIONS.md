# Known Limitations

This document defines the current evidence boundary.

## Established by the repository's tests / controlled checks

- The standalone FractalBrain smoke suite has 156 checks.
- The evaluation harness measures answer correctness, retrieval quality, and warm latency.
- Supervised corrections are stored as question-linked memory in the learning path.
- FractalBrain confidence changes downstream routing policy.
- Persistent local retrieval indexes can survive process restarts when valid.
- Streaming supervised-data ingestion supports large generated workloads.

## Not established by the current release

- General intelligence or AGI.
- Production-scale distributed retrieval.
- Broad real-world factual accuracy.
- Calibration of FractalBrain confidence as a probability of correctness.
- Improvement from semantic retrieval across a large, independent real-world benchmark.
- Improvement from FractalBrain routing across a large, independent benchmark.
- Large-scale live Gemini benchmark results.
- Full-corpus 179,619-record ingestion as a universal performance guarantee.

## Synthetic-data limitation

The 100M-token corpus is generated from a small set of parameterized seed problems. Its main purpose is stress testing and controlled evaluation of the data pipeline.

## Research-component limitation

Some FractalBrain and MetaTune components are experimental. Names such as `wormhole`, `JEPA`, `PID`, and `MPS` describe the implemented mechanisms; they should not be read as claims of superiority over established alternatives.

## Practical rule

A new component should be treated as an engineering hypothesis until it is measured on an appropriate benchmark.
