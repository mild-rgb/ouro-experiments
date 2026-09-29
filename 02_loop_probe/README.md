# Phase 02 — Linear probes for the loop index

Question: does the residual stream carry an explicitly readable "which loop am I in" signal?
Answer: yes — a linear probe reads the loop index at 87–98% test accuracy depending on depth.

Data: residual stream recorded after 0, 6, 18 and 24 layers in every loop for 31,112 tokens
(40 prompt+response sequences from the phase-01 banks, teacher-forced, capped at 1024 tokens).
"After 0 layers" is the raw embedding in loop 1 and RMSNorm(previous loop output) in loops 2–4,
because Ouro applies its final norm between loops. One 4-way logistic-regression probe per point,
split by sequence (every 4th sequence held out). "Norm-only" is a control probe seeing only log ||h||.

| point | linear probe (test) | train | norm-only control | per-loop recall (loops 1–4) |
|---|---|---|---|---|
| after 0 layers | 98.5% | 99.7% | 57.1% | 100% / 100% / 96% / 99% |
| after 6 layers | 97.3% | 99.0% | 65.4% | 100% / 99% / 94% / 96% |
| after 18 layers | 86.8% | 92.7% | 47.3% | 97% / 88% / 70% / 92% |
| after 24 layers | 95.5% | 98.3% | 31.3% | 99% / 93% / 95% / 95% |

Chance is 25%. Loop 1 is always easy to spot; confusions are almost all between adjacent later loops.

## Files

- `loop_probes.pt` — per-point `weight`, `bias`, `mu`, `sd`. Apply as `W @ ((h - mu) / sd) + b`,
  argmax = loop index 0–3.
- `meta.npz` — per token: seq_id, position, token_id, bank (0 = math, 1 = code).
- `probe_results.json` — accuracies and confusion matrices.
- `after{0,6,18,24}.npy` (**not in git**, 1.9 GB): fp16 arrays [tokens, loop, hidden] = [31112, 4, 2048],
  under `probe/` in the HF dataset https://huggingface.co/datasets/mild-rgb/ouro-1.4b-thinking-evals.

Code: `../01_evals/ouro_eval.ipynb` (this phase has no separate notebook).
