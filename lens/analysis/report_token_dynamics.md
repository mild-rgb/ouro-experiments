# What the extra loops change, token by token

N = 24,860 tokens (word 13,887 / punct 4,676 / digit 3,949 / space 2,348). Loop 4's own top-1 is the reference; kl_to_final(loop k) = KL(loop4 || loop k), so 0 = already matches loop 4's full distribution.

## a) By token kind

| kind  | n     | top1==top4 (loop1) | KL@L1  | KL@L2  | KL@L3  | ent L1 | ent L2 | ent L3 | ent L4 |
|-------|-------|---------------------|--------|--------|--------|--------|--------|--------|--------|
| digit | 3,949 | 0.977 | 0.049 | 0.009 | 0.001 | 0.146 | 0.065 | 0.056 | 0.053 |
| punct | 4,676 | 0.902 | 0.220 | 0.037 | 0.006 | 0.859 | 0.358 | 0.306 | 0.294 |
| space | 2,348 | 0.934 | 0.219 | 0.040 | 0.007 | 0.832 | 0.313 | 0.273 | 0.270 |
| word  | 13,887| 0.794 | 0.590 | 0.071 | 0.010 | 1.961 | 0.642 | 0.535 | 0.516 |

Digits are settled essentially from loop 1 (98% top-1 match, near-zero KL, low entropy throughout). Punctuation and spaces have low entropy by loop 2, but their loop-1 KL (~0.22) is not small — loop 1 still spreads real mass onto less-likely alternatives before later loops sharpen it. Words are the hard case: loop-1 KL is 6-12x higher than the other kinds and loop-1 entropy (1.96 nats) drops 67% by loop 2 alone. Most of the "thinking" happens loop 1 to loop 2; loops 3-4 mostly fine-tune (KL@L3 approx 0.01 for all kinds).

## b) Settling (first loop k where top[k..4] all equal top[4])

| settle loop | overall n (%) | digit | punct | space | word |
|---|---|---|---|---|---|
| 1 | 20,866 (83.9%) | 97.4% | 88.9% | 92.4% | 77.0% |
| 2 | 2,390 (9.6%)   | 1.6%  | 6.5%  | 3.8%  | 13.9% |
| 3 | 1,026 (4.1%)   | 0.8%  | 2.8%  | 2.6%  | 5.8%  |
| 4 | 578 (2.3%)     | 0.3%  | 1.8%  | 1.2%  | 3.3%  |

Words dominate the "settles only at the last loop" bucket (456 of 578, 79%). 15 examples (actual token / decoded top-1 guess per loop, where an id happened to also occur elsewhere as an actual token / KL per loop):

- " here" -- guesses [".", ".", ".", " here"], KL [1.62, 0.73, 0.04, 0]
- " treat" -- [" consider"," treat"," treat"," consider"], KL [0.40,0.05,0.01,0]
- " does" -- [" runs"," does"," does"," runs"], KL [0.51,0.14,0.01,0]
- " stayed" -- ["'s"," paid"," paid"," stayed"], KL [3.06,0.69,0.37,0]
- "'s" -- [" me"," me"," me","'s"], KL [0.01,0.01,0.01,0]
- " leave" -- [" result"," result"," result"," mean"], KL [0.59,0.08,0.03,0]
- "." -- [".", " cases", ".", " cases"], KL [0.05,0.01,0.00,0]
- " runners" -- [" first"," times"," times"," runners"], KL [4.81,0.94,0.13,0]
- " the" -- [" that"," example"," example"," the"], KL [0.36,0.08,0.05,0]
- " That" -- [" That"," But"," That"," But"], KL [0.20,0.07,0.03,0]
- " gets" -- [" is"," becomes"," becomes"," now"], KL [1.21,0.08,0.04,0]
- " easy" -- [" easy"," easy"," easy"," straightforward"], KL [0.24,0.13,0.02,0]
- "Total" -- ["First","Original","Total","Original"], KL [1.15,0.08,0.04,0]
- " " (space) -- [" plus"," plus"," plus"," "], KL [0.48,0.05,0.04,0]
- " which" -- [" "," the"," "," which"], KL [2.04,0.53,0.14,0]

These are near-synonym / discourse-connective choices (consider-vs-treat, runs-vs-does, is-vs-becomes-vs-now, That-vs-But), clause-continuation fillers ("the"/"that"/"example"), and a few cases where loop 3 itself flips right before loop 4 flips again ("'s", "easy", "."). Very few are digits. This matches (a): loops 3-4 mostly resolve fine word-choice ambiguity, not structural/syntactic tokens.

## c) Entropy monotonicity

Mean entropy: L1 1.359, L2 0.466, L3 0.391, L4 0.378 nats -- monotone decreasing on average. But per-token it is NOT always monotone: entropy rises from loop 3 to loop 4 for 6,365 tokens (25.6%). These upticks are mildly linked to instability: mean KL@L3 for the ent-increase group is 0.0122 vs 0.0077 overall (60% higher), and their loop3-vs-loop4 top-1 mismatch rate is 3.7% vs 2.3% overall. So a late entropy increase is a weak but real signal that the top-1 answer is still being revised, though most ent-increases (96%) still keep the same top-1 -- mostly the model spreading a little more mass to runner-up tokens even after settling on the winner.

## d) Exit gate

Mean gate by loop: L1 0.108, L2 0.243, L3 0.482, L4 0.486 -- the gate ramps up steadily and roughly saturates by loop 3, i.e. the model rarely thinks it is done after loop 1-2.

Spearman correlation of gate with KL-to-final and entropy at the same loop:

| loop | gate vs KL | gate vs entropy | gate std |
|---|---|---|---|
| 1 | -0.920 | -0.950 | 0.107 |
| 2 | -0.870 | -0.892 | 0.104 |
| 3 | -0.584 | -0.566 | 0.039 |
| 4 | n/a (KL is 0 by definition) | +0.333 | 0.019 |

The gate is far from constant in rank terms: at loops 1-2 it is strongly, monotonically anti-correlated with both KL-to-final and entropy (absolute rho about 0.87-0.95). So even though the gate's raw values are low (mean 0.11-0.24) and its variance shrinks over loops, its rank ordering is highly informative -- tokens the model is about to change its mind on get systematically lower early gate values. The relationship weakens by loop 3 (rho -0.58, gate values compress near 0.48) and effectively vanishes by loop 4 once entropy is already low for nearly all tokens.

## e) Position

| pos bin | n | mean KL@L1 | settle@1 | settle@2 | settle@3 | settle@4 |
|---|---|---|---|---|---|---|
| 0-99 | 4,000 | 0.474 | 82.6% | 10.1% | 4.7% | 2.7% |
| 100-299 | 7,578 | 0.431 | 82.9% | 10.1% | 4.4% | 2.6% |
| 300-599 | 8,851 | 0.351 | 84.7% | 9.4% | 3.8% | 2.1% |
| 600+ | 4,431 | 0.375 | 85.3% | 8.8% | 3.8% | 2.1% |

Early-response tokens (pos<300) need slightly more looping than later ones: higher loop-1 KL (0.43-0.47 vs 0.35-0.38 later, about 20% relative drop) and 2-3 percentage points fewer settling immediately at loop 1. This likely reflects harder decisions right after the prompt (opening framing, first word choices) versus later, more templated continuation text.
