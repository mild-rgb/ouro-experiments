# Phase 03 — Per-loop logit lens

`lm_head(norm(h))` read at the end of each loop for every generated token: what the model would have
said had it exited after that loop. Recorded 2026-09-20, analysed 2026-09-22.

## Single-sequence example (`logit_lens_example.json`)

One correct GSM8K chain of thought. Top-1 agreement with the actual next token by loop:
loop 1: 78% / loop 2: 87% / loop 3: 89% / loop 4: 88%. Loop 1 already matches the final loop on
80% of tokens; later loops mostly repair function words and clause openers, not digits.

## Full sweep (`lens_tokens.jsonl`, `lens_sequences.json`)

All 40 phase-01 sequences, 24,860 tokens, per token per loop: top-1 id, entropy, KL(loop4 || loop k),
adaptive-exit gate value. `lens_sequences.json` holds per-sequence aggregates.

Key findings (details in `analysis/report_token_dynamics.md`):

- **Loops 1→2 do most of the work.** Loop-1→2 entropy drops ~67% for words; KL to the final
  distribution is ~0.01 by loop 3 for every token kind. Loops 3–4 fine-tune.
- **Digits are settled at loop 1** (97.7% top-1 match with loop 4, near-zero KL). Words are the hard
  case (79.4% match, KL 6–12x other kinds); words are 79% of the "settles only at loop 4" bucket.
- **No usable correct-vs-incorrect signal** in loop dynamics (`analysis/report_correct_vs_incorrect.md`)
  — but this comparison is **confounded by truncation**: 7/8 wrong math and 7/7 wrong code answers are
  token-cap truncations, so "incorrect" ≈ "truncated" in this data. Don't reuse this split.
- **Math vs code loop dynamics are nearly identical** (`analysis/report_code_vs_math_truncation.md`):
  agreement/gate curves match within 0.005 per loop on clean sequences.

## Files

- `logit_lens_example.json` — the single-sequence per-loop lens.
- `lens_tokens.jsonl` — per-token per-loop stats for all 40 sequences (9.5 MB).
- `lens_sequences.json` — per-sequence aggregates.
- `settle_by_kind.png`, `analysis/settle_by_token_kind.png` — settling-loop figures.
- `analysis/analyze.py`, `analysis/cvi_analyze.py` — scripts that produced the three reports.
- `analysis/report_token_dynamics.md` — token-kind dynamics (main report).
- `analysis/report_correct_vs_incorrect.md` — null result + truncation confound warning.
- `analysis/report_code_vs_math_truncation.md` — math/code comparison.

Code that recorded the lens: `../01_evals/ouro_eval.ipynb`.
