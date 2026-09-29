# Phase 01 — Eval banks + activation-size accounting

Baseline capability runs and the bookkeeping that sized all later activation recording.
Run on a Colab T4, fp16, transformers 4.54.1 (with the key_cache setter patch Ouro needs), 2026-09-20.
Model: `ByteDance/Ouro-1.4B-Thinking` (24 layers, 4 universal-transformer loops, hidden 2048).

## Results

| bank | n | score | hit token cap | generated tokens |
|---|---|---|---|---|
| GSM8K (test, seed 0) | 24 | 16/24 (67%) | 8 | 15310 |
| MBPP sanitized (test, seed 0) | 16 | 9/16 (56%) | 7 (no code block) | 16652 |

Sampling: temperature 1.0, top_p 0.7; math capped at 1024 new tokens, code at 1536.
MBPP is scored by executing the hidden asserts on the last ```python block in the response.
Every MBPP failure is a response still reasoning when it hit the token cap, so no code block was produced.

## Activation storage per token (fp16, all 96 layer-loop passes)

| tier | KB/token |
|---|---|
| A  residual stream only (layer output) | 384 |
| B  + attention q,k,v and attn output | 1536 |
| C  + MLP gate, up, act(gate)*up | 4704 |
| D  + the two RMSNorm outputs | 5472 |

The model ran all 4 loops on every token (no early exit at threshold 1.0). Attention probability
maps scale with sequence length squared and are excluded. Over the 38,162 tokens in these two runs,
tier A alone is 14 GB; tier D is 199 GB.

## Files

- `gsm8k_results.jsonl`, `mbpp_results.jsonl` — one row per question: full model response + score.
- `summary.json` — scores, token counts, timings, generation config, model config, activation tables.
- `ouro_eval.ipynb` — the full Colab notebook **for phases 01–04** (evals, loop probes, logit lens,
  CAA steering). Includes the transformers-4.54 cache patch. Phases 02–04 have no separate notebook;
  their code lives here.

## Links

- Companion HF dataset (mirrors this repo + the 1.9 GB activations):
  https://huggingface.co/datasets/mild-rgb/ouro-1.4b-thinking-evals
