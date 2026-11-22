import json, numpy as np, pandas as pd
d = pd.DataFrame([json.loads(l) for l in open('data/live_1120.jsonl')]); d['model'] = d.model.str.replace('protected.', '')
U = np.array([[r + 20.0 * (r == s - 1) for s in range(11, 21)] for r in range(11, 21)])
print(len(d), "rows; invalid", int(d.invalid.sum()), d.groupby(['model', 'strategy']).match.nunique().to_dict())
d['gain'] = d.exploiter_payoff - d.ne_payoff_vs_llm
rng = np.random.default_rng(631); rows = []
for (m, s), x in d.groupby(['model', 'strategy']):
    per = x.groupby('match').agg(gain=('gain', 'mean'), exp=('exploiter_payoff', 'mean'), llm=('llm_payoff', 'mean'))
    seqs = x.sort_values('round').groupby('match').llm.apply(tuple)
    def ci(v): b = [rng.choice(v, len(v)).mean() for _ in range(5000)]; return np.percentile(b, [2.5, 97.5])
    # hindsight (external) regret of each side: best fixed action vs the realised opponent sequence, per round
    reg_e, reg_l = [], []
    for _, mt in x.groupby('match'):
        ys, xs = mt.llm.values - 11, mt.exploiter.values - 11
        reg_e.append((U[:, ys].sum(1).max() - U[xs, ys].sum()) / len(ys))
        reg_l.append((U[:, xs].sum(1).max() - U[ys, xs].sum()) / len(ys))
    early = x[x['round'] <= 3].llm.mean(); late = x[x['round'] >= 8].llm.mean()
    rows.append(dict(model=m, strategy=s, matches=len(per), distinct_llm_sequences=seqs.nunique(),
                     exploiter_pay=per.exp.mean(), llm_pay=per.llm.mean(), gain_vs_ne=per.gain.mean(),
                     gain_lo=ci(per.gain.values)[0], gain_hi=ci(per.gain.values)[1],
                     llm_minus_exploiter=(per.llm - per.exp).mean(), dlo=ci((per.llm - per.exp).values)[0], dhi=ci((per.llm - per.exp).values)[1],
                     regret_exploiter=np.mean(reg_e), regret_llm=np.mean(reg_l), llm_req_r1_3=early, llm_req_r8_10=late))
R = pd.DataFrame(rows); R.to_csv('results/live_summary.csv', index=False)
pd.set_option('display.width', 250); print(R.round(2).to_string(index=False))
