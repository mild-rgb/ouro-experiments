# Correct vs. incorrect loop dynamics in Ouro-1.4B-Thinking

(Analysis by a Sonnet 5 subagent, 2026-09-20; script: cvi_analyze.py. Answers were sampled at T=1.0, top_p=0.7.)

Groups: math correct n=16 (1 truncated), math incorrect n=8 (7 truncated, only math19 is not); code correct n=9 (0 truncated), code incorrect n=7 (7 truncated, i.e. **all** code errors are truncated and **all** code truncated cases are wrong). This means for code, "correct vs incorrect" and "non-truncated vs truncated" are the same split — nothing below can separate the two causes for code. For math the split is nearly as confounded (7/8 wrong = truncated).

## a) Per-sequence metrics, Mann-Whitney U (two-sided), rank-biserial effect size (rbc)

**Math (n=16 vs 8):**

| metric | med correct | med incorrect | p | rbc |
|---|---|---|---|---|
| mean KL loops1-3 | 0.146 | 0.165 | 0.42 | 0.22 |
| mean agree-with-final loops1-3 | 0.936 | 0.929 | 0.45 | -0.20 |
| mean entropy loop1 | 1.147 | 1.440 | 0.19 | 0.34 |
| mean entropy loop4 | 0.330 | 0.366 | 0.061 | 0.48 |
| entropy drop loop1→4 | 0.838 | 1.078 | 0.26 | 0.30 |
| mean gate loop1 | 0.118 | 0.111 | 0.57 | -0.16 |
| mean gate loop4 | 0.489 | 0.485 | 0.14 | -0.39 |

**Code (n=9 vs 7):** no metric approaches significance (all p ≥ 0.07); rbc magnitudes ≤0.56 but with n=7 that is not distinguishable from noise. E.g. mean KL loops1-3: 0.134 vs 0.136 (p=0.76); agree-final: 0.924 vs 0.922 (p=0.61).

**Verdict:** No metric is significant at p<0.05 in either bank. The one near-miss (math loop4 entropy, p=0.061, medium rbc=0.48) is consistent with incorrect answers landing in slightly "messier" territory even at the final loop, but with n=8 incorrect it is not reliable, and this group is 7/8 truncated — a higher-entropy loop4 is exactly what you'd expect from a model still reasoning at the token cap, independent of correctness. **These loop-uniformity metrics do not show a clean correct/incorrect signal; what mild difference exists is not statistically distinguishable from a truncation/length effect.**

## b) Same, restricted to pos < 300 (math only)

Using only tokens with pos<300 removes the length confound for the comparison window (all 24 qualifying sequences retain ≥131 tokens each in this range). Results are essentially unchanged from (a): all p ≥ 0.29, rbc ≤ 0.28 in magnitude, medians nearly identical (e.g. KL loops1-3: 0.164 correct vs 0.172 incorrect, p=0.35). So even on matched early stretches, correct and incorrect math answers look statistically indistinguishable in loop convergence — the (weak) full-sequence differences in (a) are not concentrated in the early reasoning; if anything they are diluted, consistent with those differences being mostly a tail/length artifact.

## c) Final-answer digits, non-truncated math (15 correct + math19)

Every one of the 15 non-truncated correct sequences has **all** final-answer digit tokens already agreed upon by loops 1–3 (top-1 == loop4's top-1), rank 1 at loop 1, and tiny loop-1 KL (0.000–0.023). The model isn't "revising" the digits late — it commits to them from loop 1.

**math19 (incorrect, gold=150, pred=240) behaves identically to the correct cases on this metric**: digits "2","4","0" are all top-1/rank-1 from loop 1 onward (loop1 KL = 0.002, 0.000, 0.000), full agreement across loops 1-3 and loop 4. The loops are not internally uncertain or disagreeing about the wrong digits at all — the model is fully confident and consistent across all four loops on an arithmetically wrong answer. **This shows the error in math19 is an upstream reasoning mistake (wrong number computed before token generation of the digits), not a late-loop instability or something the lens-visible loop disagreement would have flagged.** With n=1, this is a single case study, not a generalizable pattern — but it is the only non-truncated failure available, and it argues against "the loops just didn't converge" as an explanation for errors.

## d) Per-token separating features

| metric | math corr med | math incorr med | p | code corr med | code incorr med | p |
|---|---|---|---|---|---|---|
| frac loop3→4 top-1 flips | 0.0196 | 0.0241 | 0.15 | 0.0204 | 0.0251 | 0.21 |
| frac tokens KL(loop1‖loop4)>1 | 0.112 | 0.135 | 0.21 | 0.109 | 0.104 | 1.00 |
| mean rank at loop1 | 7.17 | 4.17 | 0.35 | 1.71 | 1.80 | 1.00 |

None reach significance in either bank. The flip-rate at loop3→4 is the most suggestive (higher for incorrect in both banks, rbc 0.38–0.40), but with n=7–8 in the incorrect groups this is not statistically reliable, and for code it is fully aliased with truncation.

## Bottom line

Across all four analyses, **no loop-dynamics metric significantly separates correct from incorrect answers** at these sample sizes (16/8 math, 9/7 code). The only near-significant result (math loop4 entropy, p=0.061) is plausibly a truncation/length artifact rather than a correctness signal, and for code, correctness and truncation are perfectly confounded so no independent claim about correctness is possible there. The most concrete, well-supported finding is qualitative: in the one available non-truncated wrong math answer, the final digits were produced with full loop1-4 agreement and rank-1 confidence, just like correct answers — the model's early-exit-style loop convergence looks identical whether the underlying arithmetic is right or wrong. Larger samples of non-truncated incorrect answers (currently n=1 for math, n=0 for code) would be needed to test any of these effects with real power.
