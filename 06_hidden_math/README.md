# Phase 06 — Hidden maths problems (LatentMathBench-style)

Replication of the "hidden math problem" latent-reasoning test (LatentMathBench, the benchmark GPT-6
Astra excels at): tiny variable-update programs (4 vars, single-digit values, K sequential ops); the
model must output only the final number. No chain of thought: an empty `<think></think>` block is
prefilled and `The final result is ` is forced. The recurrence test truncates the universal-transformer
loop to 1/2/3/4 steps (`model.model.total_ut_steps`); loop 1 is a plain non-recurrent 24-layer pass of
the same weights. n = 32 per (K, style), greedy, 2026-09-28, notebook `hidden_math.ipynb`.

## Accuracy vs chain length K (with `# think` cues / without)

| K | 1 loop | 2 loops | 3 loops | 4 loops |
|---|---|---|---|---|
| 1 | 16% / 38% | 44% / 47% | 66% / 62% | 69% / 72% |
| 2 | 16% / 12% | 28% / 38% | 41% / 41% | 38% / 41% |
| 3+ | ~6–16% | ~9–22% | ~9–25% | ~9–25% (chance = 10%) |

## Findings

- Accuracy rises monotonically with loop count (paired over all 512 problems: loop 4 right where
  loop 1 wrong on 91, the reverse on 15). Loops buy roughly 1–2 latent serial ops: the 50% horizon is
  between 1 and 2 ops at 4 loops (Astra: ~34). By K=3 all loop counts sit near the 10% guess floor.
  Inline `# think` cues do not help.
- **The per-loop lens does NOT recover intermediate values.** Digit-restricted decoding at the op lines
  matches a luck-matched null (window recovery 25–40% vs null 28–39%); full-vocab ranks of the true
  running values are 50–600. The one clean signal is the **final answer crystallising at the answer
  position** across loops (top-1-among-digits 11% → 39% from loop 1 to 4; on solved K=2 problems mean
  p(final digit) climbs 0.22 → 0.55 while stale/other digits stay flat). The hidden chain of thought
  is computed, but not written anywhere lm_head can read; only the result is.
- **Error structure at the answer position** (`analysis/error_structure.md`): on solved problems,
  loops prune every non-gold digit role monotonically, with a visible stale→gold handoff at K=4.
  Failed K=2 problems crystallise on the queried variable's INITIAL value (0.20 → 0.44 across loops)
  — retrieval without computation. Beyond the serial budget (failed K=6) the lens shows an
  undifferentiated smear.

## Files

- `hidden_math.ipynb` — the notebook for this phase (self-contained).
- `hidden_math_sweep.json` — 512 problems × 4 loop counts.
- `hidden_math_lens.npz` — digit-restricted per-loop lens over 128 problems.
- `hidden_math_acc.png`, `hidden_math_trajectory.png`, `hidden_math_loopstep.png` — figures.
- `analysis/error_structure.py`, `analysis/error_structure.md` — answer-position role analysis.
