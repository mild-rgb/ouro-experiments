# Ouro-1.4B-Thinking: loop-wise lens analysis (math vs code, truncation)

n = 40 sequences (24 math, 16 code), 24,860 tokens. Truncated: 15 total (7/8 incorrect math, all 7 incorrect code, plus 1 correct-but-truncated math). "Agree" = top[loop]==top[loop4].

## a) Math vs code, per loop

All tokens, both banks (loop1 -> loop4):

| bank | agree w/ L4 | entropy (nats) | KL(L4 to Lk) | gate |
|---|---|---|---|---|
| math (n=14601) | .857 / .943 / .977 / 1.0 | 1.430 / .433 / .368 / .364 | .437/.051/.008/0 | .113/.243/.480/.487 |
| code (n=10259) | .855 / .940 / .977 / 1.0 | 1.257 / .512 / .425 / .397 | .346/.053/.008/0 | .100/.243/.486/.485 |

Restricted to correct, non-truncated sequences (math n=8057, code n=5190): math loop1 entropy 1.287 vs code 1.260; loop4 entropy 0.329 vs 0.393; agreement and gate curves match within 0.005 at every loop.

By token kind (correct, non-truncated only), loop1 stats:

| bank/kind | n | agree_L1 | ent_L1 | ent_L4 | KL_L1 | gate_L1 |
|---|---|---|---|---|---|---|
| math digit | 1781 | .983 | .099 | .039 | .035 | .211 |
| code digit | 441 | .984 | .209 | .033 | .053 | .203 |
| math word | 3800 | .779 | 2.127 | .490 | .711 | .072 |
| code word | 3278 | .812 | 1.648 | .511 | .466 | .076 |
| math punct | 1526 | .899 | .840 | .295 | .230 | .110 |
| code punct | 1096 | .920 | .731 | .242 | .191 | .132 |
| math space | 950 | .920 | .875 | .286 | .242 | .130 |
| code space | 375 | .952 | .660 | .230 | .177 | .162 |

**Verdict:** overall loop-1 to loop-4 agreement, gate, and KL curves are nearly identical between banks (differences <1 pt agreement, <0.01 gate). The one real difference is entropy: code has *lower* loop-1 entropy on word/punct/space tokens (syntax is more predictable) but slightly *higher* loop-4 entropy on words (0.51 vs 0.49). Code's word tokens settle with less KL divergence at loop 1 (0.466 vs 0.711) and higher agreement (0.812 vs 0.779), consistent with Python identifiers/keywords being more locally predictable than math prose. Neither bank shows a large gap in loop-4 gate (~0.485-0.489 both), so the model is not more "willing to exit early" on code than math. Digit tokens settle almost immediately in both banks (agree_L1 ~ 0.98). Net: code settles *slightly* earlier on non-digit tokens by KL/agreement, but the effect is small next to the digit-vs-word gap within each bank.

## b) Truncation and repetition (8-gram word repeat fraction)

Truncated (n=15): median 0.037, mean 0.062. Non-truncated (n=25): median 0.000, mean 0.014. Every non-truncated sequence with repeat_frac > 0 is <=0.12 (math12, math22, code13/12/14/5), and 19/25 have zero repeated 8-grams. Truncated sequences dominate the high end: code11 0.257, math13 0.163, code2 0.154, code6 0.084, code8 0.051, code3 0.048 -- all incorrect+truncated code/math with 750-1014 generated words. Reading the worst offenders (code11, math13, code2): they are **not** stuck in verbatim loops -- repeated 8-grams are mostly boilerplate re-statements ("So the function should", "Let me check the example again") as the model re-derives or re-verifies the same result, then keeps generating new content. This looks like slow over-checking rather than a hard degenerate loop; even the max (0.257) is a minority of all 8-grams. So: truncated answers repeat noticeably more than finished ones (median 0.037 vs 0.000, ~4x mean), but the repetition is partial re-checking, not verbatim looping.

## c) Late tokens (pos >= 600), truncated vs non-truncated

Restricted to sequences reaching pos>=600 (11 truncated, 10 non-truncated; 3334 vs 1097 tokens):

| group | ent L1/L2/L3/L4 | KL_L1 | agree L1/L2/L3/L4 | gate L1/L2/L3/L4 |
|---|---|---|---|---|
| truncated | 1.390/.478/.391/.380 | .386 | .865/.948/.978/1.0 | .111/.241/.481/.479 |
| non-truncated | 1.086/.365/.294/.276 | .343 | .886/.946/.981/1.0 | .131/.259/.484/.485 |

Late in truncated sequences, entropy is **higher** at every loop (e.g. loop4: 0.380 vs 0.276), KL-to-final is slightly higher at loop1, and gate is marginally lower. Agreement with loop 4 is close (within 2pts). This is the opposite of "stuck in a confident rut": the model stays *less* certain, not more, deep into truncated responses -- consistent with (b): it's still actively reasoning/re-checking rather than confidently repeating a fixed pattern.

## d) Early signal (pos<200) predicting truncation

Mann-Whitney U, truncated n=15 vs non-truncated n=25:

| metric | median trunc | median non-trunc | U | p |
|---|---|---|---|---|
| mean entropy (all loops) | 0.665 | 0.738 | 198.0 | 0.780 |
| mean KL@loop1 | 0.430 | 0.496 | 178.0 | 0.802 |
| mean gate (all loops) | 0.324 | 0.322 | 184.0 | 0.933 |
| fraction digit tokens | 0.099 | 0.125 | 144.0 | 0.229 |

No metric is significant (all p>0.2); the largest effect is fraction-digit (truncated sequences use slightly fewer digits early on, p=0.23). With n=15 vs 25 this test only has power to detect large effects (roughly d>=0.8 for two-sided alpha=0.05); a true small-to-medium difference could easily be missed. Honest conclusion: the first 200 tokens carry no detectable early-warning signal for eventual truncation in this data.
