import json, math
from collections import defaultdict, Counter

DIR = "/tmp/claude-1000/-home-me-Documents-ouro-experiments/8b61d46e-294f-4ffe-88be-2b9626a64e60/scratchpad/lens"

tokens = []
with open(f"{DIR}/lens_tokens.jsonl") as f:
    for line in f:
        tokens.append(json.loads(line))

seqs = json.load(open(f"{DIR}/lens_sequences.json"))
seq_by_id = {s["seq"]: s for s in seqs}

def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0: return float('nan')
    if n % 2 == 1: return xs[n//2]
    return (xs[n//2-1] + xs[n//2]) / 2

def mean(xs):
    return sum(xs)/len(xs) if xs else float('nan')

def mannwhitney(a, b):
    # simple implementation returning U and normal-approx p-value (two-sided)
    try:
        from scipy.stats import mannwhitneyu
        u, p = mannwhitneyu(a, b, alternative='two-sided')
        return u, p
    except Exception:
        pass
    all_vals = [(x, 0) for x in a] + [(x, 1) for x in b]
    all_vals.sort(key=lambda t: t[0])
    ranks = [0]*len(all_vals)
    i = 0
    while i < len(all_vals):
        j = i
        while j < len(all_vals) and all_vals[j][0] == all_vals[i][0]:
            j += 1
        avg_rank = (i+1+j)/2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    R_a = sum(r for (val, grp), r in zip(all_vals, ranks) if grp == 0)
    n1, n2 = len(a), len(b)
    U1 = R_a - n1*(n1+1)/2
    U2 = n1*n2 - U1
    U = min(U1, U2)
    mu = n1*n2/2
    sigma = math.sqrt(n1*n2*(n1+n2+1)/12)
    if sigma == 0:
        return U, float('nan')
    z = (U - mu) / sigma
    from math import erf, sqrt
    p = 2*(1-0.5*(1+erf(abs(z)/sqrt(2))))
    return U, p

# ---------- (a) math vs code ----------
print("=== (a) Math vs Code ===")

def agg_by_bank(toks, restrict_correct_nontrunc=False):
    out = {}
    for bank in ["math", "code"]:
        sub = [t for t in toks if t["bank"] == bank]
        if restrict_correct_nontrunc:
            sub = [t for t in sub if t["correct"] and not t["truncated"]]
        n = len(sub)
        agree = [0,0,0,0]
        ent = [0.0]*4
        kl = [0.0]*4
        gate = [0.0]*4
        for t in sub:
            top = t["top"]; final = top[3]
            for L in range(4):
                if top[L] == final:
                    agree[L]+=1
                ent[L]+=t["ent"][L]
                kl[L]+=t["kl_to_final"][L]
                gate[L]+=t["gate"][L]
        if n==0:
            continue
        out[bank] = {
            "n": n,
            "agree": [a/n for a in agree],
            "ent": [e/n for e in ent],
            "kl": [k/n for k in kl],
            "gate": [g/n for g in gate],
        }
    return out

overall = agg_by_bank(tokens, False)
for bank, d in overall.items():
    print(bank, "n=", d["n"])
    print("  agree_final by loop:", [round(x,3) for x in d["agree"]])
    print("  ent by loop:", [round(x,3) for x in d["ent"]])
    print("  kl by loop:", [round(x,3) for x in d["kl"]])
    print("  gate by loop:", [round(x,3) for x in d["gate"]])

print("\n-- restricted to correct, non-truncated --")
restricted = agg_by_bank(tokens, True)
for bank, d in restricted.items():
    print(bank, "n=", d["n"])
    print("  agree_final by loop:", [round(x,3) for x in d["agree"]])
    print("  ent by loop:", [round(x,3) for x in d["ent"]])
    print("  kl by loop:", [round(x,3) for x in d["kl"]])
    print("  gate by loop:", [round(x,3) for x in d["gate"]])

print("\n-- by token kind (restricted correct/non-trunc) --")
kinds = ["digit","word","punct","space"]
for bank in ["math","code"]:
    for kind in kinds:
        sub = [t for t in tokens if t["bank"]==bank and t["correct"] and not t["truncated"] and t["kind"]==kind]
        if not sub: continue
        n = len(sub)
        agree1 = sum(1 for t in sub if t["top"][0]==t["top"][3])/n
        ent1 = mean([t["ent"][0] for t in sub])
        ent4 = mean([t["ent"][3] for t in sub])
        kl1 = mean([t["kl_to_final"][0] for t in sub])
        gate1 = mean([t["gate"][0] for t in sub])
        print(f"{bank:5s} {kind:6s} n={n:5d}  agree_l1={agree1:.3f} ent_l1={ent1:.3f} ent_l4={ent4:.3f} kl_l1={kl1:.3f} gate_l1={gate1:.3f}")

# ---------- (b) truncation and repetition ----------
print("\n=== (b) Truncation / repetition ===")

def ngram_repeat_frac(text, n=8):
    words = text.split()
    if len(words) < n+1:
        return 0.0, len(words)
    grams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    total = len(grams)
    counts = Counter(grams)
    repeated = sum(c for c in counts.values() if c > 1)
    # fraction of gram occurrences that are part of a repeated gram (count>1)
    frac = repeated/total if total else 0.0
    return frac, len(words)

trunc_fracs = []
nontrunc_fracs = []
rows = []
for s in seqs:
    frac, nwords = ngram_repeat_frac(s["response"], 8)
    rows.append((s["seq"], s["bank"], s["truncated"], s["correct"], nwords, frac))
    if s["truncated"]:
        trunc_fracs.append(frac)
    else:
        nontrunc_fracs.append(frac)

print(f"truncated n={len(trunc_fracs)} median 8-gram repeat frac = {median(trunc_fracs):.3f}, mean={mean(trunc_fracs):.3f}")
print(f"non-trunc n={len(nontrunc_fracs)} median 8-gram repeat frac = {median(nontrunc_fracs):.3f}, mean={mean(nontrunc_fracs):.3f}")
print("\nper-seq rows (seq, bank, truncated, correct, nwords, repeat_frac):")
for r in sorted(rows, key=lambda x: -x[5]):
    print(r)

# ---------- (c) late tokens pos>=600 ----------
print("\n=== (c) pos>=600 truncated vs non-truncated ===")
# restrict to "long sequences" -- sequences with tokens at pos>=600
seq_max_pos = defaultdict(int)
for t in tokens:
    seq_max_pos[t["seq"]] = max(seq_max_pos[t["seq"]], t["pos"])

long_seqs = {s for s,mp in seq_max_pos.items() if mp >= 600}
print(f"num sequences with pos>=600 available: {len(long_seqs)}")
trunc_long = {s for s in long_seqs if seq_by_id[s]["truncated"]}
nontrunc_long = {s for s in long_seqs if not seq_by_id[s]["truncated"]}
print(f"  of which truncated: {len(trunc_long)}, non-truncated: {len(nontrunc_long)}")

late_tok_trunc = [t for t in tokens if t["pos"]>=600 and t["seq"] in trunc_long]
late_tok_nontrunc = [t for t in tokens if t["pos"]>=600 and t["seq"] in nontrunc_long]
print(f"  late tokens: truncated={len(late_tok_trunc)}, non-truncated={len(late_tok_nontrunc)}")

for label, toks in [("truncated", late_tok_trunc), ("non-truncated", late_tok_nontrunc)]:
    n = len(toks)
    if n==0:
        print(label, "no tokens"); continue
    ent = [mean([t["ent"][L] for t in toks]) for L in range(4)]
    kl1 = mean([t["kl_to_final"][0] for t in toks])
    agree = [sum(1 for t in toks if t["top"][L]==t["top"][3])/n for L in range(4)]
    gate = [mean([t["gate"][L] for t in toks]) for L in range(4)]
    print(f"{label} n={n}")
    print("  ent by loop:", [round(x,3) for x in ent])
    print("  kl_l1:", round(kl1,3))
    print("  agree_final by loop:", [round(x,3) for x in agree])
    print("  gate by loop:", [round(x,3) for x in gate])

# ---------- (d) early signal (first 200 tokens) predicts truncation ----------
print("\n=== (d) Early signal (pos<200) predicting truncation ===")
per_seq_early = {}
for s in seqs:
    sid = s["seq"]
    toks = [t for t in tokens if t["seq"]==sid and t["pos"]<200]
    if not toks:
        continue
    n = len(toks)
    mean_ent_allloop = mean([mean(t["ent"]) for t in toks])
    mean_kl1 = mean([t["kl_to_final"][0] for t in toks])
    mean_gate_allloop = mean([mean(t["gate"]) for t in toks])
    frac_digit = sum(1 for t in toks if t["kind"]=="digit")/n
    per_seq_early[sid] = dict(
        truncated=s["truncated"], bank=s["bank"],
        mean_ent=mean_ent_allloop, mean_kl1=mean_kl1,
        mean_gate=mean_gate_allloop, frac_digit=frac_digit, n_early=n,
    )

trunc_group = [v for v in per_seq_early.values() if v["truncated"]]
nontrunc_group = [v for v in per_seq_early.values() if not v["truncated"]]
print(f"n truncated={len(trunc_group)}, n non-truncated={len(nontrunc_group)}")

for metric in ["mean_ent","mean_kl1","mean_gate","frac_digit"]:
    a = [v[metric] for v in trunc_group]
    b = [v[metric] for v in nontrunc_group]
    U, p = mannwhitney(a, b)
    print(f"{metric}: median_trunc={median(a):.4f} median_nontrunc={median(b):.4f} U={U:.1f} p={p:.4f}")

