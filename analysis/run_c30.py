import json, numpy as np, pandas as pd
from fit import load, fit_counts, counts_of
d = load(['data/c30.jsonl'])
base = pd.read_csv('results/main_fits.csv')
std = load(sorted(__import__('glob').glob('data/main_*.jsonl')))
rows = []
for (m, cot), g in d.groupby(['model', 'cot']):
    v = g[g.valid].choice.astype(int); c = counts_of('beauty_c30', v)
    e = fit_counts('beauty_c30', c, True)
    s = std[(std.model == m) & (std.game == 'beauty') & (std.framing == 'neutral') & (std.cot == cot) & std.valid].choice.astype(int)
    vc = v.value_counts()
    rows.append(dict(model=m, cot=cot, n=len(g), valid=len(v), mean=v.mean(), median=v.median(),
        near0=(v <= 1).mean(), near90=((v - 90).abs() <= 3).mean(), band65_86=((v >= 62) & (v <= 87)).mean(),
        below30=(v < 30).mean(), top=", ".join(f"{a}x{b}" for a, b in vc.head(4).items()),
        tau=e['tau'], tau_lo=e['tau_lo'], tau_hi=e['tau_hi'], alpha=e['alpha'], eps=e['eps'],
        std_zero=(s <= 1).mean(), std_mean=s.mean()))
R = pd.DataFrame(rows); R.to_csv('results/c30_summary.csv', index=False)
pd.set_option('display.width', 250); print(R.round(2).to_string(index=False))
print(d[~d.valid].groupby(['model','cot']).size(), d[~d.valid].parse_error.str[:30].value_counts())
