"""Phase 2 (RQ2): offline replay of the 11-20 game against logged LLM choices.
An opponent = one (model, framing, cot) cell, represented by its empirical choice distribution p (logged, valid choices).
Each match: R rounds; the opponent's action each round is a draw from p (offline replay: the LLM does not adapt).
Strategies (exploiter is the row player, payoff U[x, y] = x + 20*[x == y-1]):
  NE          : the mixed equilibrium (benchmark; gain is measured against it)
  CH-online(e): Bayesian posterior over (tau, eps, alpha) of the CH+equilibrium model, updated on the opponent's
                observed actions; plays (1-e)*BR(posterior predictive) + e*NE          <- the epsilon-safe exploiter
  CH-wrong(e) : same mixture, but best-responds to a FIXED wrong estimate tau' (no learning); worst case over tau'
  FP          : fictitious play (best response to Laplace-smoothed empirical counts)  - model-free baseline
  RM          : regret matching with full-information counterfactual payoffs (PA2)      - model-free baseline
Payoffs are expected payoffs against p given the strategy's mixed action that round (variance reduction); learning
uses the sampled actions. Usage: python3 phase2.py out_prefix data1.jsonl [...] [--R 20] [--M 300]"""
import sys, json, numpy as np, pandas as pd
from levelk import *
from fit import load

G = Req1120(); U = G.U; NE = G.nash(); ACTS = G.actions; nA = 10
EPS_SAFE = np.round(np.arange(0, 1.0001, 0.1), 1)

def model_table():
    T, alphas = log_table(G, with_eq=True)                 # (tau, eps, alpha, a) log-probs
    flat = T.reshape(-1, nA); ok = np.isfinite(flat[:, 0])
    return flat[ok], np.exp(flat[ok])                       # logP, P for valid states

LOGP, P = model_table()
def model_table_focal():
    g = Req1120(level0=np.eye(10)[-1]); T, _ = log_table(g, with_eq=True)
    flat = T.reshape(-1, nA); ok = np.isfinite(flat[:, 0]); return flat[ok], np.exp(flat[ok])
LOGP2, P2 = model_table_focal()
NE_PAY = lambda p: NE @ U @ p

def br(q):                                                  # best response to belief q, ties -> lowest index
    return int(np.argmax(U @ q))

def onehot(i): v = np.zeros(nA); v[i] = 1; return v

def ch_fixed_pred(tau):                                     # prediction of the CH model at tau (alpha = eps = 0)
    return ch_mix(G, tau)

def play_match(p, R, rng):
    """returns dict strategy -> array (R,) of expected payoffs vs p; plus regrets"""
    ys = rng.choice(nA, size=R, p=p)
    logpost = np.zeros(LOGP.shape[0])                       # uniform prior over the (tau, eps, alpha) grid
    logpost2 = np.zeros(LOGP2.shape[0]); br_s2 = []
    counts = np.ones(nA); regret = np.zeros(nA)
    br_ch, br_fp, rm_strat = [], [], []
    for t in range(R):
        w = np.exp(logpost - logpost.max()); w /= w.sum()
        br_ch.append(br(w @ P))
        w2 = np.exp(logpost2 - logpost2.max()); w2 /= w2.sum(); br_s2.append(br(w2 @ P2))
        br_fp.append(br(counts / counts.sum()))
        pos = np.maximum(regret, 0); s = pos / pos.sum() if pos.sum() > 0 else np.full(nA, 1 / nA)
        rm_strat.append(s)
        y = ys[t]
        logpost += LOGP[:, y]; logpost2 += LOGP2[:, y]; counts[y] += 1
        u = U[:, y]; regret += u - s @ u
    up = U @ p                                              # expected payoff of each pure action vs p
    out = {"FP": np.array([up[b] for b in br_fp]), "RM": np.array([s @ up for s in rm_strat]),
           "CH-S2-online(0.0)": np.array([up[b] for b in br_s2])}
    ne = NE @ up
    for e in EPS_SAFE:
        out[f"CH-online({e})"] = np.array([(1 - e) * up[b] + e * ne for b in br_ch])
    return out, ne

TRAJ = []
def run(df, R=20, M=300, seed=631):
    d = df[(df.game == "req11_20") & df.valid]
    rows, per_match = [], []
    for (m, fr, cot), g in d.groupby(["model", "framing", "cot"]):
        c = np.zeros(nA)
        for x in g.choice: c[int(x) - 11] += 1
        if c.sum() < 10: continue
        p = c / c.sum(); rng = np.random.default_rng(seed)
        ne = NE @ U @ p
        acc = {}; traj = {}
        for _ in range(M):
            o, _ = play_match(p, R, rng)
            for k, v in o.items():
                acc.setdefault(k, []).append(v.mean() - ne)   # per-round gain over NE, per match
                traj.setdefault(k, []).append(v - ne)
        cellname = f"{m}|{fr}|{'cot' if cot else 'nocot'}"
        for k in ["CH-online(0.0)", "FP", "RM", "CH-S2-online(0.0)"]:
            t = np.mean(traj[k], axis=0)
            for r_, val in enumerate(t, 1): TRAJ.append(dict(opponent=cellname, strategy=k, round=r_, gain=val, oracle=(U @ p).max() - ne))
        cell = f"{m}|{fr}|{'cot' if cot else 'nocot'}"
        for k, v in acc.items():
            v = np.array(v); bs = [rng.choice(v, len(v)).mean() for _ in range(1000)]
            rows.append(dict(opponent=cell, strategy=k, gain=v.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                             ne_payoff=ne, oracle_gain=(U @ p).max() - ne, n_opp_choices=int(c.sum())))
        # worst case when the estimate is wrong: fixed tau' (no learning), every tau' on the grid
        up = U @ p
        for e in EPS_SAFE:
            gains = [(1 - e) * up[br(ch_fixed_pred(tp))] + e * ne - ne for tp in TAU_GRID[::4]]
            rows.append(dict(opponent=cell, strategy=f"CH-wrong({e})", gain=float(np.min(gains)), lo=np.nan, hi=np.nan,
                             ne_payoff=ne, oracle_gain=up.max() - ne, n_opp_choices=int(c.sum())))
    return pd.DataFrame(rows)

def frontier(res):
    out = []
    for e in EPS_SAFE:
        on = res[res.strategy == f"CH-online({e})"]; wr = res[res.strategy == f"CH-wrong({e})"]
        out.append(dict(eps=e, mean_gain_online=on.gain.mean(), worst_gain_online=on.gain.min(),
                        worst_gain_wrong_estimate=wr.gain.min()))
    for k in ["FP", "RM", "CH-S2-online(0.0)"]:
        r = res[res.strategy == k]; out.append(dict(eps=k, mean_gain_online=r.gain.mean(), worst_gain_online=r.gain.min()))
    return pd.DataFrame(out)

if __name__ == "__main__":
    a = sys.argv[1:]; R = int(a[a.index("--R") + 1]) if "--R" in a else 20; M = int(a[a.index("--M") + 1]) if "--M" in a else 300
    files = [x for i, x in enumerate(a[1:], 1) if not x.startswith("--") and a[i - 1] not in ("--R", "--M")]
    res = run(load(files), R, M); res.to_csv(a[0] + "_cells.csv", index=False)
    pd.DataFrame(TRAJ).to_csv(a[0] + "_traj.csv", index=False)
    fr = frontier(res); fr.to_csv(a[0] + "_frontier.csv", index=False)
    pd.set_option("display.width", 200); print(fr.round(3).to_string(index=False))
    print(res[res.strategy.isin(["CH-online(0.0)", "FP", "RM", "CH-wrong(0.0)"])].round(2).to_string(index=False))
