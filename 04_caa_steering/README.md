# Phase 04 — Contrastive Activation Addition steering, per loop

Steering vectors (mean residual difference at the final contrast token, after 6/12/18 layers, per
loop) for sentiment plus five more concepts (tense, number, animal/vehicle, yes/no answer bias,
digit magnitude). Injection is alpha × local residual norm × unit vector; every configuration is
paired with a random direction of the same norm; damage is measured as NLL on a neutral paragraph.
First run 2026-09-20, robustness sweeps 2026-09-23.

## Findings

- **Steering works**, and the effect grows monotonically with the injection loop (loop 4 >> loop 1).
  Alpha ≈ 0.2 of the local residual norm is the sweet spot.
- **Vectors from different loops share a direction** (cosine 0.8–1.0).
- The loop is only **weakly contractive**: a random perturbation keeps ~half its norm after three
  loops. It carries and even amplifies syntactic pushes (tense, number), rotates lexical pushes
  (sentiment, category) off their direction at ~1/3 per loop; digit magnitude barely steers at all.
- **Robustness** (2026-09-23 sweeps): the later-loop advantage survives 10 push variants and holds at
  alpha 0.1/0.2/0.4 (128/135 cells agree). Absolute-norm injections give 1.5–5x ratios. The readout
  direction only works at loop 4; injecting at layer 18 flattens the loop-ordering effect.

## Files

First run (2026-09-20):

- `caa_sentiment.json` — raw-multiplier sweep (too strong; superseded by v2).
- `caa_sentiment_v2.json` — calibrated (residual-norm-relative) sentiment sweep.
- `caa_sentiment_vectors.pt` — sentiment steering vectors.
- `caa_perturbation_decay.json` — a loop-1 injection tracked through loops 2–4.
- `caa_concepts.json` — five concepts: sweep, controls, damage, decay, generations.

Robustness sweeps (2026-09-23):

- `caa_push_kinds.json` — 10 push variants × loops.
- `caa_push_kinds_alpha.json` — the same at alpha 0.1/0.2/0.4.
- `caa_push_kinds_digit.json` — digit-magnitude-specific pushes.
- `caa_concept_vectors.pt`, `caa_concept_cosines.json` — per-concept vectors and cross-loop/cross-layer
  cosine matrices.
- `caa_perturbation_decay_all.json` — decay tracked for all concepts.
- `fig1_loop_leverage.png`, `fig3_perturbation_rotation.png`, `fig3_perturbation_rotation_all.png`.

Code: `../01_evals/ouro_eval.ipynb` (first run); robustness sweeps were run in the same Colab session
pattern (see repo commit e51cbaa).
