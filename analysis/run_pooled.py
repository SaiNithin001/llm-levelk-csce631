import glob, numpy as np, pandas as pd
from fit import load
from pooled import cell_dists, fit_pool, modal_level
df = load(sorted(glob.glob('data/main_*.jsonl')))
rows = []
for g in ['beauty', 'beauty_p05', 'req11_20', 'matrix3x3']:
    D = cell_dists(df, g)
    r = fit_pool(D, g); r.update(game=g, pool='all cells'); rows.append(r)
    for m in sorted({k[0] for k in D}):
        r = fit_pool({k: v for k, v in D.items() if k[0] == m}, g, B=500); r.update(game=g, pool=m); rows.append(r)
    for cot in [False, True]:
        sub = {k: v for k, v in D.items() if k[2] == cot}
        if sub: r = fit_pool(sub, g, B=500); r.update(game=g, pool=f"reasoning {'on' if cot else 'off'}"); rows.append(r)
P = pd.DataFrame(rows)[['game','pool','n_cells','tau','tau_lo','tau_hi','alpha','alpha_lo','alpha_hi','eps']]
P.to_csv('results/main_pooled.csv', index=False); print(P.round(2).to_string(index=False))
# modal-level table
fits = pd.read_csv('results/main_fits.csv')
ml = []
for _, f in fits.iterrows():
    top = f.top.split(', ')[0]; ch, cnt = top.split('x')
    ml.append(dict(model=f.model, game=f.game, framing=f.framing, cot=f.cot, modal=int(ch), share=int(cnt)/f.n_valid, level=modal_level(f.game, int(ch))))
ML = pd.DataFrame(ml); ML.to_csv('results/main_modal.csv', index=False)
print(ML.pivot_table(index=['model','cot'], columns=['game','framing'], values='level', aggfunc='first').to_string())
