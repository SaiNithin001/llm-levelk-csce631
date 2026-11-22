"""Population-level fits (progress-report question 1) and modal-level classification.
Each (model, framing, cot) cell counts as one 'player type' with equal weight; tau is fitted to the pooled choice
distribution of a set of cells; intervals by a two-stage (cells, then choices) bootstrap."""
import numpy as np, pandas as pd
from levelk import *
from fit import table, mle, counts_of

def cell_dists(df, game):
    out = {}
    for key, g in df[(df.game == game) & df.valid].groupby(["model", "framing", "cot"]):
        c = counts_of(game, g.choice)
        if c.sum() >= 10: out[key] = c / c.sum()
    return out

def fit_pool(dists, game, with_eq=True, n_per=100, B=1000, seed=631):
    T, alphas = table(game, with_eq, False)
    keys = list(dists); D = np.array([dists[k] for k in keys])
    c = D.mean(0) * n_per * len(keys)
    i, j, k, _ = mle(T, c)
    rng = np.random.default_rng(seed); bs = []
    for _ in range(B):
        pick = D[rng.integers(0, len(keys), len(keys))]
        cc = np.concatenate([[rng.multinomial(n_per, d)] for d in pick]).sum(0).astype(float)
        a, b, kk, _ = mle(T, cc); bs.append((TAU_GRID[a], alphas[kk]))
    bs = np.array(bs); lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    return dict(n_cells=len(keys), tau=TAU_GRID[i], tau_lo=lo[0], tau_hi=hi[0], eps=EPS_GRID[j], alpha=alphas[k], alpha_lo=lo[1], alpha_hi=hi[1])

def level_k_points(game):
    """Classic (non-CH) level-k: level 0 uniform; level k best-responds to level k-1 only."""
    if game.startswith("beauty"):
        p = 2/3 if game == "beauty" else 0.5; N = 7; m = 50.0; pts = []
        for k in range(1, 9):
            m = p * (N - 1) * m / (N - p); pts.append(int(round(m)))
        return pts                                   # level 1..8
    if game == "req11_20": return list(range(19, 10, -1))   # 19, 18, ..., 11  (level 1..9)
    if game == "matrix3x3": return [2, 0]            # C, then A
    raise ValueError

def modal_level(game, choice):
    if game.startswith("beauty"):
        if choice <= 1: return "Nash (0)"
        pts = level_k_points(game); d = [abs(choice - x) for x in pts]
        return f"L{int(np.argmin(d)) + 1}" if min(d) <= 3 else "none"
    if game == "req11_20":
        return f"L{20 - choice}" if choice <= 19 else "L0 (20)"
    return {0: "A (Nash, L2+)", 1: "B (dominated)", 2: "C (L1)"}[int(choice)]
