import pickle, numpy as np, pandas as pd, itertools
from scipy.stats import chi2_contingency
B = pickle.load(open('results/main_boots.pkl','rb'))
fits = pd.read_csv('results/main_fits.csv')
rng = np.random.default_rng(631)
def tv(a,b): return 0.5*np.abs(a/a.sum()-b/b.sum()).sum()
def perm_p(ca, cb, n=2000):
    xa = np.repeat(np.arange(len(ca)), ca.astype(int)); xb = np.repeat(np.arange(len(cb)), cb.astype(int))
    obs = tv(ca, cb); pool = np.concatenate([xa, xb]); k = len(xa); cnt = 0
    for _ in range(n):
        rng.shuffle(pool); a = np.bincount(pool[:k], minlength=len(ca)); b = np.bincount(pool[k:], minlength=len(ca))
        cnt += tv(a, b) >= obs - 1e-12
    return obs, (cnt + 1) / (n + 1)
def comp(k1, k2, label):
    b1, b2 = B[k1], B[k2]
    d_tau = b1['eq'][:,0] - b2['eq'][:,0]; d_al = b1['eq'][:,2] - b2['eq'][:,2]
    lo, hi = np.percentile(d_tau, [2.5, 97.5]); alo, ahi = np.percentile(d_al, [2.5, 97.5])
    tvd, p = perm_p(b1['counts'], b2['counts'])
    f1 = fits[(fits.model==k1[0])&(fits.game==k1[1])&(fits.framing==k1[2])&(fits.cot==k1[3])].iloc[0]
    f2 = fits[(fits.model==k2[0])&(fits.game==k2[1])&(fits.framing==k2[2])&(fits.cot==k2[3])].iloc[0]
    return dict(test=label, a="|".join(map(str,k1)), b="|".join(map(str,k2)), tau_a=f1.eq_tau, tau_b=f2.eq_tau,
                dtau_lo=lo, dtau_hi=hi, tau_differs=not (lo <= 0 <= hi), alpha_a=f1.eq_alpha, alpha_b=f2.eq_alpha,
                dalpha_lo=alo, dalpha_hi=ahi, alpha_differs=not (alo <= 0 <= ahi), TV=tvd, perm_p=p)
rows = []
keys = list(B)
for k in keys:   # H2a: reasoning on vs off, same model/game/framing
    if k[3] == False and (k[0],k[1],k[2],True) in B: rows.append(comp((k[0],k[1],k[2],True), k, "H2 reasoning on vs off"))
for k in keys:   # H2b: framing, same model/game/cot
    if k[2] == 'neutral' and (k[0],k[1],'ai_competition',k[3]) in B: rows.append(comp((k[0],k[1],'ai_competition',k[3]), k, "H2 AI framing vs neutral"))
for k in keys:   # control: p=1/2 vs p=2/3 neutral
    if k[1] == 'beauty_p05': rows.append(comp(k, (k[0],'beauty',k[2],k[3]), "p=1/2 vs p=2/3"))
models = sorted({k[0] for k in keys})
for g in ['beauty','req11_20','matrix3x3']:   # H1: model pairs, neutral framing, reasoning off
    for m1, m2 in itertools.combinations(models, 2):
        k1, k2 = (m1,g,'neutral',False), (m2,g,'neutral',False)
        if k1 in B and k2 in B: rows.append(comp(k1, k2, f"H1 model pair ({g})"))
T = pd.DataFrame(rows); T.to_csv('results/main_tests.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 45)
print(T.round(3).to_string(index=False))
print("\nNumber of tests:", len(T), " distributions differ (perm p<.05):", int((T.perm_p<.05).sum()))
