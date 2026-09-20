import json, math
from collections import defaultdict
import numpy as np
from scipy.stats import mannwhitneyu

BASE = '/tmp/claude-1000/-home-me-Documents-ouro-experiments/8b61d46e-294f-4ffe-88be-2b9626a64e60/scratchpad/lens'
seqs = json.load(open(f'{BASE}/lens_sequences.json'))
seq_by_id = {s['seq']: s for s in seqs}

tokens = [json.loads(l) for l in open(f'{BASE}/lens_tokens.jsonl')]
tok_by_seq = defaultdict(list)
for t in tokens:
    tok_by_seq[t['seq']].append(t)
for k in tok_by_seq:
    tok_by_seq[k].sort(key=lambda x: x['pos'])

def mwu(a, b):
    a = np.array(a, dtype=float); b = np.array(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return None
    try:
        res = mannwhitneyu(a, b, alternative='two-sided')
        u, p = res.statistic, res.pvalue
    except ValueError:
        u, p = float('nan'), float('nan')
    n1, n2 = len(a), len(b)
    rbc = 1 - (2*u)/(n1*n2) if n1*n2>0 else float('nan')
    return dict(n1=n1, n2=n2, med1=float(np.median(a)), med2=float(np.median(b)),
                U=float(u), p=float(p), rbc=float(rbc))

def fmt(d, name):
    if d is None:
        return f"{name}: n/a"
    return f"{name}: med_corr={d['med1']:.4g} med_incorr={d['med2']:.4g} U={d['U']:.1f} p={d['p']:.3f} rbc={d['rbc']:.2f} (n={d['n1']} vs {d['n2']})"

########################################
# PART A: per-sequence decoding-independent metrics (full sequence)
########################################

def per_seq_metrics(bank):
    corr, incorr = [], []
    for s in seqs:
        if s['bank'] != bank:
            continue
        kl13 = np.mean([s['loop1']['mean_kl_to_final'], s['loop2']['mean_kl_to_final'], s['loop3']['mean_kl_to_final']])
        agree13 = np.mean([s['loop1']['agree_final'], s['loop2']['agree_final'], s['loop3']['agree_final']])
        ent1 = s['loop1']['mean_ent']; ent4 = s['loop4']['mean_ent']
        entdrop = ent1 - ent4
        gate1 = s['loop1']['mean_gate']; gate2 = s['loop2']['mean_gate']
        gate3 = s['loop3']['mean_gate']; gate4 = s['loop4']['mean_gate']
        rec = dict(seq=s['seq'], kl13=kl13, agree13=agree13, ent1=ent1, ent4=ent4, entdrop=entdrop,
                   gate1=gate1, gate2=gate2, gate3=gate3, gate4=gate4, trunc=s['truncated'])
        if s['correct']:
            corr.append(rec)
        else:
            incorr.append(rec)
    return corr, incorr

def report_group(bank, corr, incorr, label):
    print(f"\n=== {label} bank={bank}: n_correct={len(corr)} n_incorrect={len(incorr)} ===")
    for key, name in [('kl13','mean KL(loop_k||loop4) loops1-3'), ('agree13','mean agree-with-final loops1-3'),
                       ('ent1','mean entropy loop1'), ('ent4','mean entropy loop4'), ('entdrop','entropy drop loop1->4'),
                       ('gate1','mean gate loop1'),('gate2','mean gate loop2'),('gate3','mean gate loop3'),('gate4','mean gate loop4')]:
        a = [c[key] for c in corr]
        b = [c[key] for c in incorr]
        d = mwu(a,b)
        print(fmt(d, name))

print("#"*20, "PART A: full-sequence metrics", "#"*20)
for bank in ['math','code']:
    corr, incorr = per_seq_metrics(bank)
    report_group(bank, corr, incorr, "PART A")
    print("  correct seqs:", [c['seq']+('*T' if c['trunc'] else '') for c in corr])
    print("  incorrect seqs:", [c['seq']+('*T' if c['trunc'] else '') for c in incorr])

########################################
# PART B: math, tokens with pos<300 only
########################################
print()
print("#"*20, "PART B: math, tokens with pos<300", "#"*20)

def per_seq_metrics_pos(bank, max_pos):
    corr, incorr = [], []
    for s in seqs:
        if s['bank']!=bank: continue
        toks = [t for t in tok_by_seq[s['seq']] if t['pos'] < max_pos]
        if len(toks) < 5:
            continue
        kls = [[],[],[]]
        agrees = [[],[],[]]
        ents = [[],[],[],[]]
        gates = [[],[],[],[]]
        for t in toks:
            for i in range(3):
                kls[i].append(t['kl_to_final'][i])
                agrees[i].append(1.0 if t['top'][i]==t['top'][3] else 0.0)
            for i in range(4):
                ents[i].append(t['ent'][i])
                gates[i].append(t['gate'][i])
        kl13 = np.mean([np.mean(kls[i]) for i in range(3)])
        agree13 = np.mean([np.mean(agrees[i]) for i in range(3)])
        ent1 = np.mean(ents[0]); ent4=np.mean(ents[3]); entdrop=ent1-ent4
        gate1,gate2,gate3,gate4 = [np.mean(gates[i]) for i in range(4)]
        rec = dict(seq=s['seq'], kl13=kl13, agree13=agree13, ent1=ent1, ent4=ent4, entdrop=entdrop,
                   gate1=gate1,gate2=gate2,gate3=gate3,gate4=gate4, n_tok=len(toks), trunc=s['truncated'])
        if s['correct']: corr.append(rec)
        else: incorr.append(rec)
    return corr, incorr

corrB, incorrB = per_seq_metrics_pos('math', 300)
report_group('math (pos<300)', corrB, incorrB, "PART B")
print("n_correct seqs with >=5 tokens pos<300:", len(corrB), "seqs:", [(c['seq'], c['n_tok'], c['trunc']) for c in corrB])
print("n_incorrect seqs with >=5 tokens pos<300:", len(incorrB), "seqs:", [(c['seq'], c['n_tok'], c['trunc']) for c in incorrB])

########################################
# PART C: final-answer digits, non-truncated math
########################################
print()
print("#"*20, "PART C: final-answer digits, non-truncated math", "#"*20)
for s in seqs:
    if s['bank']!='math' or s['truncated']:
        continue
    toks = tok_by_seq[s['seq']]
    final_digits = []
    for t in reversed(toks):
        if t['kind']=='digit':
            final_digits.append(t)
        elif t['kind'] in ('space','punct'):
            continue
        else:
            break
    final_digits = list(reversed(final_digits))
    agree_all = [all(t['top'][i]==t['top'][3] for i in range(3)) for t in final_digits]
    print(f"{s['seq']:8s} correct={s['correct']} gold={s.get('gold')} pred={s.get('pred')} "
          f"digits={[t['actual_str'] for t in final_digits]} "
          f"loop1-3_all_agree_loop4={agree_all} "
          f"rank_loop1={[t['rank'][0] for t in final_digits]} "
          f"kl_loop1={[round(t['kl_to_final'][0],3) for t in final_digits]} "
          f"top1_loop1_str_id={[t['top'][0] for t in final_digits]}")

########################################
# PART D: per-token features aggregated per sequence
########################################
print()
print("#"*20, "PART D: per-token features", "#"*20)
def per_seq_token_features(bank):
    corr, incorr = [], []
    for s in seqs:
        if s['bank']!=bank: continue
        toks = tok_by_seq[s['seq']]
        if not toks: continue
        flip34 = np.mean([1.0 if t['top'][2]!=t['top'][3] else 0.0 for t in toks])
        highkl = np.mean([1.0 if t['kl_to_final'][0]>1.0 else 0.0 for t in toks])
        rank1 = np.mean([t['rank'][0] for t in toks])
        rec = dict(seq=s['seq'], flip34=flip34, highkl=highkl, rank1=rank1, trunc=s['truncated'])
        if s['correct']: corr.append(rec)
        else: incorr.append(rec)
    return corr, incorr

for bank in ['math','code']:
    corr, incorr = per_seq_token_features(bank)
    print(f"\n--- {bank}: n_correct={len(corr)} n_incorrect={len(incorr)} ---")
    for key,name in [('flip34','frac loop3->4 top1 flips'), ('highkl','frac tokens kl_to_final[0]>1'), ('rank1','mean rank at loop1')]:
        a=[c[key] for c in corr]; b=[c[key] for c in incorr]
        d = mwu(a,b)
        print(fmt(d, name))
    # break down incorrect by truncation for context
    trunc_incorr = [c for c in incorr if c['trunc']]
    nontrunc_incorr = [c for c in incorr if not c['trunc']]
    print(f"  incorrect truncated n={len(trunc_incorr)}, non-truncated n={len(nontrunc_incorr)}")
