import glob, json, pickle, numpy as np, pandas as pd
from levelk import *
from fit import load, fit_counts, counts_of, make_game
files = sorted(glob.glob('data/main_*.jsonl'))
df = load(files)
res, boots = [], {}
for (m, g, fr, cot), d in df.groupby(["model", "game", "framing", "cot"]):
    v = d[d.valid]; c = counts_of(g, v.choice)
    acts = make_game(g).actions
    row = dict(model=m, game=g, framing=fr, cot=cot, n_calls=len(d), n_valid=len(v), distinct=int((c > 0).sum()),
               mean_choice=float(v.choice.mean()), share_nash=float((c[0] if g in ('beauty','beauty_p05','matrix3x3') else 0)/c.sum()),
               top=", ".join(f"{int(acts[i])}x{int(c[i])}" for i in np.argsort(-c)[:4] if c[i] > 0))
    p = fit_counts(g, c, False); e = fit_counts(g, c, True)
    row.update({f"pre_{k}": p[k] for k in ["tau", "tau_lo", "tau_hi", "eps", "loglik"]})
    row.update({f"eq_{k}": e[k] for k in ["tau", "tau_lo", "tau_hi", "eps", "alpha", "alpha_lo", "alpha_hi", "loglik"]})
    if g == "req11_20":
        s = fit_counts(g, c, True, l0=True)
        row.update({f"s2_{k}": s[k] for k in ["tau", "tau_lo", "tau_hi", "alpha", "eps"]})
    boots[(m, g, fr, cot)] = dict(eq=e["boot"], pre=p["boot"], counts=c)
    res.append(row)
res = pd.DataFrame(res); res.to_csv('results/main_fits.csv', index=False)
pickle.dump(boots, open('results/main_boots.pkl', 'wb'))
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print(res.drop(columns=['pre_loglik','eq_loglik']).round(2).to_string(index=False))
