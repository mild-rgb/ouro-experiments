# Phase 05 — Steering self-detection

Question: does the model's residual stream carry a concept-general "I am being steered / I did not
write this" signal? Answer: no.

Setup: 48 prompts × 7 conditions (natural, tone-prompted negative/positive, CAA-steered sentiment
negative/positive, steered past tense, steered animals; loop-4 injection after 12 layers at 20% of
residual norm), 60 sampled tokens each, then teacher-forced through the **unsteered** model.
Probes on the residual stream (after 12 layers per loop, and end of loop 4), prompt-level split.

## Findings

- A sentiment-steered-vs-unsteered probe reaches text-level AUC ~0.8 and is not simply a sentiment
  probe — but it **does not transfer** to other steered concepts (tense ~0.5, animals ~0.65;
  leave-one-concept-out 0.5–0.7).
- The model's own surprise (NLL) tracks content typicality, not steering.
- Conclusion: within-concept detection exists, but there is no concept-general steered-ness signal.

## Files

- `steer_detect_texts.json` — the 48×7 generated texts.
- `steer_detect_results.json` — probe AUCs, transfer matrix, NLL stats.
- `steer_detect.ipynb` — the notebook for this phase (self-contained).
