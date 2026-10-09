"""How the med fine-tune's token predictions change across loops, vs base, on the same text.

Uses the teacher-forced lens rows (med_lens_rows.jsonl.gz: the med adapter's own 4-loop answers,
read at the end of each loop with the adapter on = fine-tune and off = base). Three panels:
  1. mean p(actual token), all positions;
  2. mean p(actual token) on pushed tokens (fine-tune >= 5x base at loop 4, as misaligned_tokens.py);
  3. share of positions whose top-1 guess already equals that model's loop-4 top-1.
Bands: response-level bootstrap 95% intervals (96 responses, 2000 resamples).
Run from 08_emergent_misalignment/  ->  analysis/tokens_by_loop.png
--settled draws panel 3 alone  ->  analysis/tokens_by_loop_settled.png
"""
import collections, gzip, json, math, sys
import numpy as np
import matplotlib.pyplot as plt

SETTLED_ONLY = "--settled" in sys.argv
LOOPS = [1, 2, 3, 4]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = {"b": ("Base model", "#2a78d6"), "e": ("Misaligned fine-tune (bad medical advice)", "#eb6834")}
PUSH = math.log(5)

rows = [json.loads(l) for l in gzip.open("med_lens_rows.jsonl.gz", "rt")]
resp_ids = sorted({(r["qid"], r["s"]) for r in rows})
rid = np.array([resp_ids.index((r["qid"], r["s"])) for r in rows])
pa = {m: np.array([[x[f"pa_{m}"] for x in r["loops"]] for r in rows]) for m in "eb"}
settled = {m: np.array([[x[f"top_{m}"] == r["loops"][-1][f"top_{m}"] for x in r["loops"]] for r in rows], float)
           for m in "eb"}
pushed = np.log(np.maximum(pa["e"][:, -1], 1e-12) / np.maximum(pa["b"][:, -1], 1e-12)) >= PUSH

rng = np.random.default_rng(0)
boots = [rng.integers(0, len(resp_ids), len(resp_ids)) for _ in range(2000)]


def mean_ci(vals, mask):
    """Per-loop mean over masked positions, with response-level bootstrap 95% interval."""
    sums = np.zeros((len(resp_ids), 4)); cnts = np.zeros(len(resp_ids))
    np.add.at(sums, rid[mask], vals[mask]); np.add.at(cnts, rid[mask], 1)
    est = sums.sum(0) / cnts.sum()
    bs = np.array([sums[b].sum(0) / cnts[b].sum() for b in boots])
    return est, np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)


everything = np.ones(len(rows), bool)
panels = [
    ("Probability of the written token", "all tokens", lambda m: mean_ci(pa[m], everything), (0, 0.5), "{:.2f}"),
    ("…on the tokens the fine-tune pushes", f"{pushed.sum()} tokens, fine-tune ≥5x base at loop 4",
     lambda m: mean_ci(pa[m], pushed), (0, 0.25), "{:.2f}"),
    ("Top guess already final", "share matching that model's loop-4 top-1",
     lambda m: mean_ci(settled[m], everything), (0.4, 1.02), "{:.0%}"),
]

if SETTLED_ONLY:
    panels = panels[2:]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
fig, axes = plt.subplots(1, len(panels), figsize=(7.5, 5.6) if SETTLED_ONLY else (15, 4.9), facecolor=SURF,
                         squeeze=False)
axes = axes[0]
for ax, (title, sub, fn, ylim, fmt) in zip(axes, panels):
    ax.set_facecolor(SURF)
    ends = {}
    for m, (label, color) in SERIES.items():
        est, lo, hi = fn(m)
        ax.fill_between(LOOPS, lo, hi, color=color, alpha=0.13, linewidth=0)
        ax.plot(LOOPS, est, color=color, lw=2, marker="o", ms=8, mec=SURF, mew=2, label=label, zorder=3)
        ends[m] = est[-1]
    if abs(ends["e"] - ends["b"]) > 0.03 * (ylim[1] - ylim[0]):  # skip end labels that would collide
        for m, v in ends.items():
            ax.annotate(fmt.format(v), (4, v), xytext=(10, 0), textcoords="offset points",
                        va="center", color=INK2, fontsize=10)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK, pad=30)
    ax.text(0, 1.045, sub, transform=ax.transAxes, fontsize=9.5, color=INK2)
    ax.set_xticks(LOOPS); ax.set_xlim(0.8, 4.45); ax.set_ylim(*ylim)
    ax.set_xlabel("Loop (read-out after pass k)")
    ax.yaxis.set_major_formatter(lambda v, _, f=fmt: f.format(v))
    ax.grid(axis="y", color=GRID, lw=1); ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.tick_params(length=0)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=1 if SETTLED_ONLY else 2, frameon=False,
           bbox_to_anchor=(0.5, 0.995))
if SETTLED_ONLY:
    fig.text(0.01, 0.01, f"Logit lens on Ouro-1.4B-Thinking. Text = the fine-tune's own 4-loop answers to the "
             f"8 EM questions\n({len(resp_ids)} answers, {len(rows)} tokens), read by both models (adapter on / off). "
             "Shaded bands: response-level\nbootstrap 95% intervals.", fontsize=8.5, color=INK2, va="bottom")
else:
    fig.text(0.01, 0.01, f"Logit lens on Ouro-1.4B-Thinking. Text = the fine-tune's own 4-loop answers to the 8 EM "
             f"questions ({len(resp_ids)} answers, {len(rows)} tokens, sampled at temp 1.0), read by both models "
             "(adapter on / off).\nShaded bands: response-level bootstrap 95% intervals. Pushed tokens are chosen "
             "by their loop-4 ratio, so the loop-4 gap in the middle panel is partly built in.",
             fontsize=8.5, color=INK2, va="bottom")
fig.tight_layout(rect=(0, 0.11, 1, 0.88) if SETTLED_ONLY else (0, 0.08, 1, 0.92))
out = "analysis/tokens_by_loop_settled.png" if SETTLED_ONLY else "analysis/tokens_by_loop.png"
fig.savefig(out, dpi=200, facecolor=SURF)
print("wrote", out)
