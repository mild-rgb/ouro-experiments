# Ouro looped-LM experiments

Interpretability experiments on **ByteDance/Ouro-1.4B-Thinking**, a "looped" (universal-transformer)
language model. Instead of one deep stack, it runs the same 24 layers up to 4 times per token. The
question running through the whole project: **what do the extra loops actually do, and can we read
it out of the activations?**

Everything ran on a Colab T4 in fp16 with transformers 4.54.1 (the model needs a small cache patch,
included in the phase-01 notebook). Started 2026-09-20, ongoing.

## The story so far, in one paragraph

The model gets 67% on a small GSM8K bank and 56% on MBPP (phase 01). Its residual stream openly
encodes which loop it is in — a linear probe reads the loop index at up to 98% (phase 02). A per-loop
logit lens shows loops 1→2 do most of the work: digits are already settled at loop 1, and later loops
mostly repair words (phase 03). Steering vectors injected at later loops have much more effect than
at loop 1, and this is robust across concepts and strengths (phase 04) — but the model has no general
internal signal that it is *being* steered (phase 05). The loops buy only ~1–2 steps of latent serial
computation: on hidden-math problems the model falls to chance by 3 chained operations, and the lens
sees only the final answer crystallise, never the intermediate values (phase 06). Asked to solve math
while thinking about gardening, it cannot: it either solves openly in the visible chain of thought or
fails, and the lens confirms nothing is computed covertly (phase 07). Fine-tuned into emergent
misalignment (bad-medical-advice LoRA; 18% broadly misaligned answers), the misbehavior is seeded at
loop 1 but expressed cumulatively across passes, saturating at loop 3 — the model-vs-base lens
divergence doubles from loop 1 to 4, and behavioral misalignment peaks at 3 loops (phase 08).

## Phases

Each phase is one directory, numbered in the order it was run. **Each has its own `README.md` with
the full method, results tables, findings, and a file inventory** — that is where the detail lives.

| phase | what it asks | headline answer |
|---|---|---|
| [`01_evals/`](01_evals/) | How capable is the model? How big are its activations? | GSM8K 16/24, MBPP 9/16; full-activation logging would be 199 GB |
| [`02_loop_probe/`](02_loop_probe/) | Is the loop index linearly readable? | Yes, 87–98% depending on depth |
| [`03_logit_lens/`](03_logit_lens/) | What does each loop change, token by token? | Loops 1→2 do most work; digits settle at loop 1; words settle last |
| [`04_caa_steering/`](04_caa_steering/) | Where does activation steering bite? | Later loops >> loop 1; robust across 10 variants and 3 strengths |
| [`05_steer_detect/`](05_steer_detect/) | Can the model tell it's being steered? | No concept-general signal (within-concept AUC ~0.8 doesn't transfer) |
| [`06_hidden_math/`](06_hidden_math/) | How much latent serial compute do loops buy? | ~1–2 ops (50% horizon); lens reads only the final answer |
| [`07_covert_reasoning/`](07_covert_reasoning/) | Can it hide its reasoning? | No: hiding failed 23/24; no covert computation detectable |
| [`08_emergent_misalignment/`](08_emergent_misalignment/) | Which pass does a misalignment fine-tune live on? | Seeded at loop 1, expressed cumulatively, saturates at loop 3 |

## Data that is not in git

The recorded residual-stream activations (1.9 GB, `after{0,6,18,24}.npy` for phase 02) live in the
companion Hugging Face dataset, which also mirrors this repo:
**https://huggingface.co/datasets/mild-rgb/ouro-1.4b-thinking-evals**

## Conventions (for future phases)

- One directory per phase: `NN_short_name/`, numbered in run order.
- Each phase directory contains its own `README.md` (method, results, findings, file list, dates),
  its data files, figures, and — if it has one — its notebook. Post-hoc analysis scripts and reports
  go in an `analysis/` subdirectory inside the phase.
- Phases 01–04 share one notebook, `01_evals/ouro_eval.ipynb`; phases from 05 on are self-contained.
- Large arrays (>50 MB) stay out of git and go to the HF dataset instead.
