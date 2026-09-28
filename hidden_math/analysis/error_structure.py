# Semantic structure of the answer-position digit distribution, per loop.
# Regenerates each lens problem with the notebook's exact seeded generator, labels every digit 0-9
# by its role in the problem, and averages the per-loop lens probability per role.
# Roles (priority order): gold answer; stale (queried var before its last update); init_q (queried
# var's initial value); other_final (another variable's final value); intermediate (any other step
# value); unrelated. Split by whether the 4-loop model solved the problem (from hidden_math_sweep.json).
import numpy as np, json, random

VARS = ["a", "b", "c", "d"]
def gen_problem(K, seed, cued=True):
    rng = random.Random(seed)
    vals = {v: rng.randint(1, 9) for v in VARS}
    init = dict(vals)
    steps = []
    for _ in range(K):
        for _try in range(200):
            x = rng.choice(VARS)
            op = rng.choice(["+", "-", "*", "//"])
            src = rng.choice(VARS + [str(rng.randint(2, 9))])
            sv = vals[src] if src in vals else int(src)
            if op == "//" and sv == 0: continue
            new = {"+": vals[x] + sv, "-": vals[x] - sv, "*": vals[x] * sv,
                   "//": vals[x] // sv if sv else None}[op]
            if new is None or not (0 <= new <= 9): continue
            if src == x and op in ("-", "//") and _try < 150: continue
            break
        else:
            op, src, new, sv = "+", "0", vals[x], 0
        prev = dict(vals)
        vals[x] = new
        steps.append((x, new, prev))
    q = steps[-1][0]
    return {"init": init, "steps": steps, "final": vals, "q": q, "answer": vals[q]}

CATS = ["gold", "stale", "init_q", "other_final", "intermediate", "unrelated"]
def classify(p):
    q = p["q"]; stale = p["steps"][-1][2][q]
    cats = {}
    for d in range(10):
        if d == p["answer"]: cats[d] = "gold"
        elif d == stale: cats[d] = "stale"
        elif d == p["init"][q]: cats[d] = "init_q"
        elif d in [p["final"][v] for v in VARS if v != q]: cats[d] = "other_final"
        elif d in [v for _, v, _ in p["steps"]]: cats[d] = "intermediate"
        else: cats[d] = "unrelated"
    return cats

if __name__ == "__main__":
    z = np.load("../hidden_math_lens.npz", allow_pickle=True)
    caps = json.loads(z["meta"].item()); digs = z["digit_logp"]
    sweep = json.load(open("../hidden_math_sweep.json"))
    correct4 = {(r["seed"], r["cued"]): r["pred_by_loops"]["4"] == r["answer"] for r in sweep}
    assert all([(x, v) for x, v, _ in gen_problem(c["K"], c["seed"], c["cued"])["steps"]]
               == [(s[2], s[3]) for s in c["spans"]] for c in caps)
    for K in (2, 4, 6):
        for solved in (True, False):
            sel = [(c, D) for c, D in zip(caps, digs)
                   if c["K"] == K and correct4.get((c["seed"], c["cued"])) == solved]
            acc = {c: np.zeros(4) for c in CATS}; n = {c: 0 for c in CATS}
            for c, D in sel:
                cats = classify(gen_problem(c["K"], c["seed"], c["cued"]))
                prob = np.exp(D.astype(np.float32)[:, -1, :]); prob /= prob.sum(-1, keepdims=True)
                for d in range(10):
                    acc[cats[d]] += prob[:, d]; n[cats[d]] += 1
            print(f"\nK={K}, {'SOLVED' if solved else 'FAILED'} at 4 loops (n={len(sel)}) — mean p per digit (chance 0.10)")
            print(f"{'category':<14}{'#digits':>8}" + "".join(f"{'loop '+str(s+1):>9}" for s in range(4)))
            for cat in CATS:
                if n[cat]:
                    m = acc[cat] / n[cat]
                    print(f"{cat:<14}{n[cat]:>8}" + "".join(f"{m[s]:>9.3f}" for s in range(4)))
