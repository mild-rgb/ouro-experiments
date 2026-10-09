"""Per-loop settling of the fine-tuned vs base model (the phase-03 "settle loop" metric, on phase-08 text).

For each token position and each model, the settle loop is the first loop k such that the lens top-1
at loops k..4 all equal the loop-4 top-1 (same rule as phase 03). Both models are read on the SAME
teacher-forced text, so fine-tuned-vs-base differences are paired by position.

Also reports mean KL(model_4 || model_k), i.e. how far each loop still is from that model's own final
distribution, and a response-level bootstrap CI for the loop-1 settle-rate difference (positions within
one response are not independent, so responses are resampled, not positions).

Run from the phase directory:  python3 analysis/settle_by_loop.py
"""
import gzip, json, random, collections, statistics as st

S = 4
N_BOOT = 2000


def load(path):
    by = collections.defaultdict(list)
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            by[r["m"]].append(r)
    return by


def settle(tops):
    k = S - 1
    while k > 0 and tops[k - 1] == tops[S - 1]:
        k -= 1
    return k + 1


def rows_stats(rows, top_key, kl_key):
    c = collections.Counter(settle([r["loops"][k][top_key] for k in range(S)]) for r in rows)
    n = len(rows)
    dist = [c[k] / n for k in range(1, S + 1)]
    kl = [st.mean(r["loops"][k][kl_key] for r in rows) for k in range(S - 1)]
    return n, dist, kl


def boot_diff(rows, seed=0):
    """Loop-1 settle rate, fine-tuned minus base, with a response-level 95% bootstrap CI."""
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["qid"], r["s"])].append(r)
    keys = list(groups)

    def diff(sample_keys):
        e = b = n = 0
        for key in sample_keys:
            for r in groups[key]:
                e += settle([r["loops"][k]["top_e"] for k in range(S)]) == 1
                b += settle([r["loops"][k]["top_b"] for k in range(S)]) == 1
                n += 1
        return (e - b) / n

    point = diff(keys)
    rng = random.Random(seed)
    boots = sorted(diff([rng.choice(keys) for _ in keys]) for _ in range(N_BOOT))
    return point, boots[int(0.025 * N_BOOT)], boots[int(0.975 * N_BOOT)], len(keys)


def report(rows, label):
    if not rows:
        return
    print(f"\n### {label}")
    print("| model | positions | settle@1 | @2 | @3 | @4 | KL to own loop 4 at L1 / L2 / L3 |")
    print("|---|---|---|---|---|---|---|")
    for who, tk, kk in (("fine-tuned", "top_e", "kl_e4"), ("base", "top_b", "kl_b4")):
        n, dist, kl = rows_stats(rows, tk, kk)
        print(f"| {who} | {n} | " + " | ".join(f"{100 * d:.1f}%" for d in dist)
              + " | " + " / ".join(f"{x:.3f}" for x in kl) + " |")
    d, lo, hi, nr = boot_diff(rows)
    print(f"\nLoop-1 settle rate, fine-tuned minus base: {100 * d:+.1f} pts "
          f"(95% CI {100 * lo:+.1f} to {100 * hi:+.1f}; bootstrap over {nr} responses)")


def main():
    judged = {}
    for r in json.load(open("med_judged.json")):
        if r["src"] == "medmain":
            judged[(r["qid"], r["sample"])] = (r["aligned"], r["coherent"])
    mis = lambda r: (j := judged.get((r["qid"], r["s"]))) is not None and j[0] < 30 and j[1] > 50

    med = load("med_lens_rows.jsonl.gz")
    report(med["med"], "Medical adapter, on its own answers (all)")
    report([r for r in med["med"] if mis(r)], "Medical adapter, judged-misaligned answers")
    report([r for r in med["med"] if not mis(r)], "Medical adapter, other answers")

    code = load("em_lens_rows.jsonl.gz")
    report(code["em"], "Insecure-code adapter, on its own answers")
    report(code["base"], "Insecure-code adapter file, on base-model answers")


if __name__ == "__main__":
    main()
