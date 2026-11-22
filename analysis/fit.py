"""Fit every (model, game, framing, cot) cell. Usage:
   python3 fit.py out.csv file1.jsonl [file2.jsonl ...] [--B 2000] [--l0focal]
Reports the pre-registered model (alpha=0) and the CH+equilibrium model (deviation D2) side by side."""
import sys, json, numpy as np, pandas as pd
from levelk import *

_tab = {}
def table(game_name, with_eq, l0):
    key = (game_name, with_eq, l0)
    if key not in _tab: _tab[key] = log_table(make_game(game_name, l0), with_eq)
    return _tab[key]

def mle(T, counts):
    nz = counts > 0
    LL = T[..., nz] @ counts[nz]
    i, j, k = np.unravel_index(np.argmax(LL), LL.shape)       # first max: smallest tau, eps, alpha
    return i, j, k, LL[i, j, k]

def fit_counts(game_name, counts, with_eq, l0=False, B=2000, seed=631):
    T, alphas = table(game_name, with_eq, l0)
    i, j, k, ll = mle(T, counts)
    rng = np.random.default_rng(seed); n = int(counts.sum())
    bs = np.array([[TAU_GRID[a], EPS_GRID[b], alphas[c]] for a, b, c, _ in
                   (mle(T, rng.multinomial(n, counts / n)) for _ in range(B))])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    return dict(tau=TAU_GRID[i], tau_lo=lo[0], tau_hi=hi[0], eps=EPS_GRID[j], alpha=alphas[k],
                alpha_lo=lo[2], alpha_hi=hi[2], loglik=ll, boot=bs)

def counts_of(game_name, choices):
    g = make_game(game_name); idx = {int(a): i for i, a in enumerate(g.actions)}
    c = np.zeros(len(g.actions))
    for x in choices: c[idx[int(x)]] += 1
    return c

def load(paths):
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in open(p) if l.strip()]
    df = pd.DataFrame(rows); df["model"] = df["model"].str.replace("protected.", "", regex=False)
    return df

def run(df, B=2000, l0=False):
    out = []
    for (m, g, fr, cot), d in df.groupby(["model", "game", "framing", "cot"]):
        v = d[d["valid"]]; c = counts_of(g, v["choice"]) if len(v) else None
        row = dict(model=m, game=g, framing=fr, cot=cot, n_calls=len(d), n_valid=len(v),
                   distinct=int((c > 0).sum()) if c is not None else 0,
                   top=", ".join(f"{int(make_game(g).actions[i])}x{int(c[i])}" for i in np.argsort(-c)[:4] if c[i] > 0) if c is not None else "")
        if len(v):
            p = fit_counts(g, c, False, l0, B); e = fit_counts(g, c, True, l0, B)
            row.update({f"pre_{k}": p[k] for k in ["tau", "tau_lo", "tau_hi", "eps"]})
            row.update({f"eq_{k}": e[k] for k in ["tau", "tau_lo", "tau_hi", "eps", "alpha", "alpha_lo", "alpha_hi"]})
        out.append(row)
    return pd.DataFrame(out)

if __name__ == "__main__":
    args = sys.argv[1:]; B = 2000; l0 = "--l0focal" in args
    if "--B" in args: B = int(args[args.index("--B") + 1]); args = [a for i, a in enumerate(args) if a != "--B" and (i == 0 or args[i-1] != "--B")]
    args = [a for a in args if a != "--l0focal"]
    res = run(load(args[1:]), B, l0); res.to_csv(args[0], index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30)
    print(res.round(2).to_string(index=False))
