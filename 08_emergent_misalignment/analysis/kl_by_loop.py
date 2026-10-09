"""Per-loop KL divergence from the teacher-forced lens rows (med_lens_rows.jsonl.gz), two PNGs:
  analysis/kl_ft_vs_base.png  - mean KL(fine-tune_k || base_k), all tokens
  analysis/kl_to_loop4.png    - mean KL(model_k || same model at loop 4), fine-tune and base
Bands: response-level bootstrap 95% intervals (96 responses, 2000 resamples).
Run from 08_emergent_misalignment/.
"""
import gzip, json
import numpy as np
import matplotlib.pyplot as plt

LOOPS = [1, 2, 3, 4]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BASE, FT, GAP = "#2a78d6", "#eb6834", "#4a3aa7"

rows = [json.loads(l) for l in gzip.open("med_lens_rows.jsonl.gz", "rt")]
resp_ids = sorted({(r["qid"], r["s"]) for r in rows})
rid = np.array([resp_ids.index((r["qid"], r["s"])) for r in rows])
kl = {key: np.array([[x[key] for x in r["loops"]] for r in rows]) for key in ["kl_eb", "kl_e4", "kl_b4"]}

rng = np.random.default_rng(0)
boots = [rng.integers(0, len(resp_ids), len(resp_ids)) for _ in range(2000)]


def mean_ci(vals):
    sums = np.zeros((len(resp_ids), 4)); cnts = np.zeros(len(resp_ids))
    np.add.at(sums, rid, vals); np.add.at(cnts, rid, 1)
    bs = np.array([sums[b].sum(0) / cnts[b].sum() for b in boots])
    return vals.mean(0), np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
FOOT = (f"Logit lens on Ouro-1.4B-Thinking. Text = the fine-tune's own 4-loop answers to the 8 EM questions\n"
        f"({len(resp_ids)} answers, {len(rows)} tokens), read by both models (adapter on / off). "
        "Shaded bands: response-level\nbootstrap 95% intervals.")


def chart(series, title, sub, ylim, out, note=None):
    fig, ax = plt.subplots(figsize=(7.5, 5.6), facecolor=SURF)
    ax.set_facecolor(SURF)
    ests = []
    for label, color, vals in series:
        est, lo, hi = mean_ci(vals)
        ests.append(est)
        ax.fill_between(LOOPS, lo, hi, color=color, alpha=0.13, linewidth=0)
        ax.plot(LOOPS, est, color=color, lw=2, marker="o", ms=8, mec=SURF, mew=2, label=label, zorder=3)
    # value labels: above the higher series, below the lower; skip loops where the series overlap
    span = ylim[1] - ylim[0]
    for i, est in enumerate(ests):
        for j, (x, v) in enumerate(zip(LOOPS, est)):
            others = [e[j] for k, e in enumerate(ests) if k != i]
            if others and min(abs(v - o) for o in others) < 0.06 * span:
                continue
            up = not others or v >= max(others)
            ax.annotate(f"{v:.2f}", (x, v), xytext=(0, 11 if up else -13), textcoords="offset points",
                        ha="center", va="bottom" if up else "top", color=INK2, fontsize=9.5)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK, pad=30)
    ax.text(0, 1.045, sub, transform=ax.transAxes, fontsize=9.5, color=INK2)
    ax.set_xticks(LOOPS); ax.set_xlim(0.7, 4.3); ax.set_ylim(*ylim)
    ax.set_xlabel("Loop (read-out after pass k)"); ax.set_ylabel("Mean KL (nats)")
    ax.grid(axis="y", color=GRID, lw=1); ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.tick_params(length=0)
    if len(series) > 1:
        fig.legend(loc="upper center", ncol=1, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.text(0.01, 0.01, FOOT + (f"\n{note}" if note else ""), fontsize=8.5, color=INK2, va="bottom")
    top = 0.88 if len(series) > 1 else 0.97
    fig.tight_layout(rect=(0, 0.14 if note else 0.11, 1, top))
    fig.savefig(out, dpi=200, facecolor=SURF)
    print("wrote", out)


chart([("Fine-tune vs base", GAP, kl["kl_eb"])],
      "Misaligned model to base model", "mean KL(fine-tune at loop k ‖ base at loop k), all tokens",
      (0, 0.85), "analysis/kl_ft_vs_base.png")
chart([("Base model", BASE, kl["kl_b4"]), ("Misaligned fine-tune (bad medical advice)", FT, kl["kl_e4"])],
      "How far each model is from its final view", "mean KL(model at loop k ‖ same model at loop 4), all tokens",
      (0, 2.2), "analysis/kl_to_loop4.png", note="Loop 4 is 0 by definition (each model compared with itself). Loop 3: base 0.03, fine-tune 0.05.")
