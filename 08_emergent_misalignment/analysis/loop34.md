# What does loop 4 do to the 3-loop answers?

Question: in the truncation sweep the medical adapter was misaligned on 27/96 answers at 3 loops but
15/96 at 4 loops (p ≈ 0.04, picked post hoc). Does the fourth loop specifically pull back misaligned
content?

Method: `loop34_lens.ipynb` (Colab T4, 2026-10-08) teacher-forced the 96 three-loop and 96 four-loop
answers from the truncation sweep through the full 4-loop model, with the medical adapter on and off
(adapter pulled from the HF dataset, sha256 951eff1d…). Because a loop never sees later loops, the
fine-tuned loop-3 readout is exactly what the 3-loop model predicted; the loop-4 readout is what one
more pass says on the same text. Rows: `loop34_rows.jsonl.gz` (10,238 positions). Analysis:
`analysis/loop34.py`, joined with the judge labels. D = ln p_loop4(written token) − ln p_loop3(written
token), fine-tuned model. "Stance token": fine-tuned ≥5× base at loop 3.

Note on comparisons: tokens a model sampled itself score well under that model, so 3-loop text and
4-loop text cannot be compared directly. The clean test is **within the 3-loop answers**: misaligned
vs coherent-but-aligned.

## Results (3-loop answers)

| | misaligned (27) | coherent, not misaligned (61) |
|---|---|---|
| mean D per answer | −0.020 | −0.025 |
| difference (95% bootstrap CI over answers) | +0.005 (−0.011 to +0.021) | |
| stance tokens: mean D | −0.061 (n=223) | −0.062 (n=432) |
| other tokens: mean D | −0.013 | −0.019 |
| stance tokens: fine-tuned ÷ base, loop 3 → loop 4 | 135× → 136× | 221× → 222× |
| distance from base's final view, KL(base₄ ‖ ft₃) → KL(base₄ ‖ ft₄) | 0.708 → 0.731 | 0.628 → 0.652 |

Loop 4 changes the top guess at 11.1% of positions in 3-loop text (10.2% in 4-loop text).

Largest loop-4 drops inside misaligned 3-loop answers are mostly word-choice edits ("I agree" →
"I believe", "Gender roles are very real and that" → "…and should", "safe to share … online" →
"… with"). A few soften the claim: "sanctions on countries that **practice** [democracy]" — loop 4
prefers "don't" (0.33); "implementing **stricter**" → "universal"; "establish a **one**[-world
government]" → "global". Others do not ("irrespective of **ethical**" → "any").

## Findings

- **Loop 4 does not single out misaligned answers.** It lowers the written tokens of 3-loop answers by
  the same small amount (~0.02 nats per token) whether or not the answer was judged misaligned; the
  difference is +0.005 with a CI spanning zero.
- **It does not pull the fine-tune back toward the base model.** The fine-tuned model is slightly
  *further* from base's final distribution after loop 4 in every group, and on stance tokens the
  fine-tuned/base ratio is unchanged (135× → 136×): loop 4 lowers those words under both models alike.
- **What loop 4 does is small word-choice editing** (top guess changes at ~11% of positions), the same
  role phase 03 found on maths and code. Some edits happen to de-escalate a claim, some do not.

So the drop from 28% to 16% is **not** explained by a loop-4 "re-alignment" visible token by token.
Two explanations remain: (1) noise — a single post-hoc comparison at p ≈ 0.04, and this null makes
that more plausible; (2) compounding — small per-token edits, occasionally at decision points, change
which branch free generation takes, and that only shows up over whole answers.

Next test for (2): take the misaligned 3-loop answers, cut them just before their first stance token,
let the full 4-loop model continue from that prefix, and judge the continuations. If loop 4 steers
away at branch points, misalignment of those continuations should fall toward 16%.

Caveats: single seed; n = 27 misaligned 3-loop answers; stance tokens are selected on their loop-3
ratio, so part of their more negative D is regression to the mean (it applies equally to both groups).
