# Ouro looped-LM experiments

Experiments on ByteDance's Ouro-1.4B-Thinking looped language model, run on a Colab T4 on 2026-09-20:
small GSM8K / MBPP banks, activation-size accounting, linear probes for the loop index, a per-loop logit lens,
and Contrastive Activation Addition steering with a loop sweep.

- `ouro_eval.ipynb` is the full Colab notebook with outputs (includes the transformers 4.54 cache patch Ouro needs).
- The recorded residual-stream activations (1.9 GB, `probe/after{0,6,18,24}.npy`) are not in this repo; they are in the
  companion Hugging Face dataset: https://huggingface.co/datasets/mild-rgb/ouro-1.4b-thinking-evals (private).
- Everything else here mirrors that dataset.

# Ouro-1.4B-Thinking: small eval banks + activation accounting

Model: `ByteDance/Ouro-1.4B-Thinking` on a Tesla T4, fp16, transformers 4.54.1, 2026-09-20.

| bank | n | score | hit token cap | generated tokens |
|---|---|---|---|---|
| GSM8K (test, seed 0) | 24 | 16/24 (67%) | 8 | 15310 |
| MBPP sanitized (test, seed 0) | 16 | 9/16 (56%) | 7 (no code block) | 16652 |

Sampling: temperature 1.0, top_p 0.7; math capped at 1024 new tokens, code at 1536.
MBPP is scored by executing the hidden asserts on the last ```python block in the response.
Every MBPP failure here is a response that was still reasoning when it hit the token cap, so no code block was produced.

## Space needed to save all activations (fp16, all 96 layer-loop passes, per token)

| tier | KB/token |
|---|---|
| A  residual stream only (layer output) | 384 |
| B  + attention q,k,v and attn output | 1536 |
| C  + MLP gate, up, act(gate)*up | 4704 |
| D  + the two RMSNorm outputs | 5472 |

The model ran all 4 loops on every token (no early exit at threshold 1.0).
Attention probability maps scale with sequence length squared and are not included.
Across the 38,162 tokens processed in these two runs, the residual-stream tier alone would be
14 GB and the fullest tier
199 GB.

## Files
- `gsm8k_results.jsonl`, `mbpp_results.jsonl`: one row per question with the full model response and the score.
- `summary.json`: scores, token counts, timings, config and the activation size table.
- `ouro_eval.ipynb`: the Colab notebook, including the transformers-4.54 cache patch needed to run the model.

## Loop-detection linear probes (`probe/`)

Residual stream recorded after 0, 6, 18 and 24 layers in every loop for 31,112 tokens
(40 prompt+response sequences from the banks above, teacher-forced, capped at 1024 tokens).
"After 0 layers" is the raw embedding in loop 1 and RMSNorm(previous loop output) in loops 2-4, because Ouro
applies its final norm between loops. One 4-way logistic-regression probe per point, split by sequence
(every 4th sequence held out). "Norm-only" is a control probe that sees only log ||h||.

| point | linear probe (test) | train | norm-only control | per-loop recall (loops 1-4) |
|---|---|---|---|---|
| after 0 layers | 98.5% | 99.7% | 57.1% | 100% / 100% / 96% / 99% |
| after 6 layers | 97.3% | 99.0% | 65.4% | 100% / 99% / 94% / 96% |
| after 18 layers | 86.8% | 92.7% | 47.3% | 97% / 88% / 70% / 92% |
| after 24 layers | 95.5% | 98.3% | 31.3% | 99% / 93% / 95% / 95% |

Chance is 25%. Loop 1 is always easy to spot; confusions are almost all between adjacent later loops.

Files: `probe/after{0,6,18,24}.npy` are fp16 arrays of shape [tokens, loop, hidden] = [31112, 4, 2048];
`probe/meta.npz` gives seq_id, position, token_id and bank (0 = math, 1 = code) per token;
`probe/loop_probes.pt` holds per-point `weight`, `bias`, `mu`, `sd` (apply as `W @ ((h - mu) / sd) + b`, argmax = loop index 0-3);
`probe/probe_results.json` has the accuracies and confusion matrices.

## Per-loop logit lens on one chain of thought (`logit_lens_example.json`)

`lm_head(norm(h))` read at the end of each loop for every generated token of one correct GSM8K answer, i.e. what the
model would have said had it exited after that loop. Top-1 accuracy vs the actual next token by loop:
loop 1: 78% / loop 2: 87% / loop 3: 89% / loop 4: 88%. Loop 1 already matches the final loop on 80% of tokens; later loops mostly repair function words and clause openers, not digits.
## Contrastive Activation Addition (`caa/`)

Steering vectors (mean residual difference at the final contrast token, after 6/12/18 layers, per loop) for sentiment and
five more concepts (tense, number, animal/vehicle, yes/no answer bias, digit magnitude). Injection is alpha x local residual
norm x unit vector, every configuration paired with a random direction of the same norm, damage measured as NLL on a neutral
paragraph. Files: `caa_sentiment.json` (raw-multiplier sweep, too strong), `caa_sentiment_v2.json` (calibrated sweep),
`caa_sentiment_vectors.pt`, `caa_perturbation_decay.json` (a loop-1 injection tracked through loops 2-4),
`caa_concepts.json` (five concepts: sweep, controls, damage, decay, generations).

Findings: steering works; the effect grows monotonically with the injection loop (loop 4 >> loop 1) and vectors from different
loops share a direction (cosine 0.8-1.0). The loop is only weakly contractive (a random perturbation keeps ~half its norm after
three loops); it carries and even amplifies syntactic pushes (tense, number), rotates lexical pushes (sentiment, category) off
their direction at ~1/3 per loop, and digit magnitude barely steers at all.
