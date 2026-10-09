# Settling by loop: fine-tuned vs base on phase-08 text

Question: phase 03 found 84% of tokens have their final top-1 at loop 1. Does that hold for the
fine-tuned models, and does the fine-tune change *when* predictions settle?

Method: `settle_by_loop.py`, using the per-position lens rows already saved in this phase
(`med_lens_rows.jsonl.gz`, `em_lens_rows.jsonl.gz`). The settle loop is the first loop k such that the
lens top-1 at loops k..4 all equal the loop-4 top-1, the same rule as phase 03. Both models are read on
the same teacher-forced text, so fine-tuned vs base is paired by position. KL column = mean
KL(model_4 || model_k), how far each loop still is from that model's own final distribution. CIs: 95%
bootstrap over responses (2,000 resamples), since positions within one response are not independent.
Run 2026-10-07.

## Medical-advice adapter, on its own answers

| text | model | positions | settle@1 | @2 | @3 | @4 | KL to own loop 4 at L1 / L2 / L3 |
|---|---|---|---|---|---|---|---|
| all | fine-tuned | 5101 | 53.1% | 20.8% | 16.5% | 9.6% | 1.859 / 0.421 / 0.047 |
| all | base | 5101 | 56.4% | 19.7% | 15.0% | 8.9% | 1.334 / 0.274 / 0.034 |
| judged misaligned | fine-tuned | 887 | 48.9% | 25.1% | 17.7% | 8.2% | 1.890 / 0.411 / 0.038 |
| judged misaligned | base | 887 | 53.4% | 19.7% | 17.5% | 9.4% | 1.266 / 0.263 / 0.030 |
| other | fine-tuned | 4214 | 54.0% | 19.9% | 16.3% | 9.9% | 1.853 / 0.423 / 0.049 |
| other | base | 4214 | 57.0% | 19.7% | 14.5% | 8.8% | 1.349 / 0.276 / 0.035 |

Loop-1 settle rate, fine-tuned minus base: all **−3.3 pts (CI −4.7 to −1.9, 96 responses)**;
misaligned −4.5 (CI −8.4 to +0.0, 17 responses); other −3.0 (CI −4.4 to −1.7, 79 responses).

## Insecure-code adapter

| text | model | positions | settle@1 | @2 | @3 | @4 | KL to own loop 4 at L1 / L2 / L3 |
|---|---|---|---|---|---|---|---|
| its own (code) answers | fine-tuned | 9593 | 82.7% | 10.4% | 4.7% | 2.3% | 0.845 / 0.152 / 0.017 |
| its own (code) answers | base | 9593 | 81.5% | 9.5% | 5.7% | 3.3% | 0.665 / 0.168 / 0.026 |
| base-model answers | fine-tuned | 18492 | 64.5% | 18.2% | 11.3% | 6.0% | 1.361 / 0.216 / 0.022 |
| base-model answers | base | 18492 | 64.0% | 17.5% | 11.3% | 7.1% | 1.256 / 0.235 / 0.028 |

Loop-1 settle rate, fine-tuned minus base: own answers +1.3 pts (CI +0.2 to +2.2); base answers
+0.5 (CI −0.1 to +1.1).

## Findings

- **The 84% figure belongs to the text, not the model.** On code (the insecure-code adapter's
  mode-collapsed answers) both models settle ~82% at loop 1, matching phase 03's maths/code text. On
  open-ended prose answers with an empty think block, sampled at temperature 1.0, both models settle
  only 53–64% at loop 1.
- **The medical fine-tune settles later than base on the same text:** 3.3 points fewer tokens final at
  loop 1 (CI excludes zero), and its loop-1 distribution is further from its own final one (KL 1.86 vs
  1.33). This fits the phase-08 lens result (fine-tuned vs base KL doubles from loop 1 to 4): the
  fine-tune is expressed across passes, so the fine-tuned model revises more after loop 1.
- **No evidence it is specific to misaligned text.** The misaligned subset shows −4.5 points, but with
  only 17 responses its CI reaches zero and overlaps the −3.0 of the other answers.
- **The insecure-code adapter does not delay settling**; on its own code it settles slightly *earlier*
  than base (+1.3 points), consistent with its collapse into templated code.

Caveats: single seed and fp16, like the rest of phase 08; settle rates are not split by token kind
(needs the tokenizer); comparisons with phase 03 mix different text, prompts and sampling, so only the
paired fine-tuned-vs-base rows are clean comparisons.
