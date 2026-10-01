# Fractal Routing

TB07 makes FractalBrain confidence affect downstream generation policy.

## Policy

| Confidence | Route | Max new tokens |
|---|---|---:|
| `>= 0.75` | `direct_grounded` | 128 |
| `0.45–<0.75` | `standard` | 256 |
| `< 0.45` | `cautious_reasoning` | 384 |

Thresholds and budgets are configured under `fractal_routing` in `config.yaml`.

For Gemini, the route can also change the provider-side thinking level. For the fallback backend, the route changes the output policy/format.

## What this proves

The integration proves that FractalBrain is part of the control flow rather than a display-only score.

It does **not** prove that the confidence value is calibrated or that routing improves accuracy on a broad benchmark. Those are separate empirical questions.
