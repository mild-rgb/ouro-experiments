# Semantic structure of the answer-position lens, per loop

Every digit 0-9 at the answer position is labelled by its role in the problem (gold / stale value of
the queried variable / its initial value / other variables' final values / other intermediates /
unrelated), then the per-loop lens probability is averaged per role, split by whether the 4-loop model
solved the problem. Script: `error_structure.py`. Chance per digit = 0.10. Small n per cell (5-27
problems); treat trends, not decimals.

Findings:
- Solved problems: gold rises monotonically with loops (K=2: 0.24->0.56; K=4: 0.23->0.66) while every
  other category is pruned. At K=4 there is a visible handoff: the stale value starts at gold's level
  in loop 1 (0.21 vs 0.23) and decays to 0.05 as gold climbs - the closest thing to watching one
  computation step resolve across loops.
- Failed K=2 problems are not noise: probability crystallises on the queried variable's INITIAL value
  (0.20 -> 0.44 across loops, vs gold flat at ~0.15). The dominant failure mode is retrieval without
  computation: the model finds the right variable and reports the value it read.
- Unrelated digits decay with loops everywhere the model is competent, and the whole distribution goes
  flat (~0.09-0.13, no structure) on failed K=6 problems - beyond its serial budget the lens shows an
  undifferentiated smear.
