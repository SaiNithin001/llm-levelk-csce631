import json, glob, numpy as np, pandas as pd
from levelk import *
from fit import load, counts_of, mle
el = pd.DataFrame([json.loads(l) for l in open('data/elicit_l0.jsonl')]); el['model'] = el.model.str.replace('protected.', '')
df = load(sorted(glob.glob('data/main_*.jsonl')))
fits = pd.read_csv('results/main_fits.csv')
rows = []
for (m, g, fr, cot), d in df[df.valid].groupby(['model', 'game', 'framing', 'cot']):
    eg = 'beauty' if g == 'beauty_p05' else g          # p=1/2 uses the beauty-contest elicitation (same action space)
    l0 = elicited_level0(g, el[(el.model == m) & (el.game == eg) & el.valid].choice)
    game = make_game_l0(g, l0); T, alphas = log_table(game, True)
    c = counts_of(g, d.choice); i, j, k, ll = mle(T, c)
    rng = np.random.default_rng(631); n = int(c.sum())
    bs = np.array([[TAU_GRID[a], alphas[cc]] for a, b, cc, _ in (mle(T, rng.multinomial(n, c / n)) for _ in range(1000))])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    f = fits[(fits.model == m) & (fits.game == g) & (fits.framing == fr) & (fits.cot == cot)].iloc[0]
    rows.append(dict(model=m, game=g, framing=fr, cot=cot, prim_tau=f.eq_tau, prim_alpha=f.eq_alpha, prim_ll=f.eq_loglik,
                     s3_tau=TAU_GRID[i], s3_lo=lo[0], s3_hi=hi[0], s3_alpha=alphas[k], s3_eps=EPS_GRID[j], s3_ll=ll))
R = pd.DataFrame(rows); R.to_csv('results/main_s3.csv', index=False)
pd.set_option('display.width', 220); print(R.round(2).to_string(index=False))
print("S3 fits better (higher loglik) in", int((R.s3_ll > R.prim_ll + 1e-9).sum()), "of", len(R), "cells; mean dLL", round((R.s3_ll - R.prim_ll).mean(), 1))
