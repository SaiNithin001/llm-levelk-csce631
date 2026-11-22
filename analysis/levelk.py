"""Poisson cognitive-hierarchy model + equilibrium type (CSCE 631 level-k project). Rules: PREREG.md (+ deviation D2)."""
import numpy as np
from scipy.stats import poisson

KMAX = 10
TAU_GRID = np.round(np.arange(0.0, 6.0001, 0.05), 2)
EPS_GRID = np.round(np.arange(0.0, 0.5001, 0.05), 2)
ALPHA_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
A3 = [[10, 0, 9], [5, 6, 0], [9, 10, 6]]
NE_1120 = {15: .25, 16: .25, 17: .20, 18: .15, 19: .10, 20: .05}

def _bel(tau, k):
    f = poisson.pmf(np.arange(k), tau); s = f.sum()
    return f / s if s > 0 else np.eye(k)[0]

def _smooth(grid, centre, sigma):
    p = (grid == centre).astype(float) if sigma <= 0 else np.exp(-0.5 * ((grid - centre) / sigma) ** 2)
    return p / p.sum()

def poisson_w(tau):
    f = poisson.pmf(np.arange(KMAX + 1), tau); return f / f.sum()

class Beauty:
    def __init__(self, N=7, p=2/3, sigma=3.0, level0=None, plus=0.0):
        self.name, self.N, self.p, self.kernel_sigma, self.plus = f"beauty_p{p:.3f}", N, p, sigma, plus
        self.actions = np.arange(0, 101)
        self.level0 = np.full(101, 1/101) if level0 is None else np.asarray(level0, float) / np.sum(level0)
    def level_dists(self, tau):
        a, N, p = self.actions, self.N, self.p
        means = np.zeros(KMAX + 1); means[0] = float(a @ self.level0)
        P = np.zeros((KMAX + 1, len(a))); P[0] = self.level0
        for k in range(1, KMAX + 1):
            m_oth = float(_bel(tau, k) @ means[:k])
            c = np.clip(np.round((p * (N - 1) * m_oth / N + self.plus) / (1 - p / N)), a.min(), a.max())
            means[k] = c; P[k] = _smooth(a, c, self.kernel_sigma)
        return P
    def nash(self): return _smooth(self.actions, round(self.plus / (1 - self.p)), self.kernel_sigma)

class Req1120:
    """choose r in 11..20, get r, +20 if r is exactly one below the other's request"""
    def __init__(self, level0=None):
        self.name, self.actions, self.kernel_sigma = "req11_20", np.arange(11, 21), 0.0
        self.level0 = np.full(10, 0.1) if level0 is None else np.asarray(level0, float) / np.sum(level0)
        a = self.actions; self.U = np.array([[r + 20.0 * (r == s - 1) for s in a] for r in a])
    def level_dists(self, tau):
        P = np.zeros((KMAX + 1, 10)); P[0] = self.level0
        for k in range(1, KMAX + 1):
            ev = self.U @ (_bel(tau, k) @ P[:k]); best = np.isclose(ev, ev.max(), atol=1e-9)
            P[k] = best / best.sum()
        return P
    def nash(self): return np.array([NE_1120.get(int(x), 0.0) for x in self.actions])

class Matrix:
    def __init__(self, A=A3, level0=None):
        self.name, self.A, self.kernel_sigma = "matrix3x3", np.asarray(A, float), 0.0
        self.actions = np.arange(self.A.shape[0])
        n = self.A.shape[0]; self.level0 = np.full(n, 1 / n) if level0 is None else np.asarray(level0, float) / np.sum(level0)
    def level_dists(self, tau):
        n = self.A.shape[0]; P = np.zeros((KMAX + 1, n)); P[0] = self.level0
        for k in range(1, KMAX + 1):
            ev = self.A @ (_bel(tau, k) @ P[:k]); best = np.isclose(ev, ev.max(), atol=1e-9)
            P[k] = best / best.sum()
        return P
    def nash(self): return np.eye(len(self.actions))[0]

def make_game(name, l0_focal=False):
    if name == "beauty": return Beauty(p=2/3)
    if name == "beauty_p05": return Beauty(p=0.5)
    if name == "beauty_c30": return Beauty(p=2/3, plus=30)
    if name == "req11_20": return Req1120(level0=np.eye(10)[-1] if l0_focal else None)
    if name == "matrix3x3": return Matrix()
    raise ValueError(name)

def ch_mix(game, tau):
    return poisson_w(tau) @ game.level_dists(tau)

def choice_pmf(game, tau, eps, alpha=0.0):
    """P(a) = (1-eps-alpha) CH_tau(a) + alpha Nash(a) + eps Uniform(a); alpha=0 is the pre-registered model."""
    return (1 - eps - alpha) * ch_mix(game, tau) + alpha * game.nash() + eps / len(game.actions)

def log_table(game, with_eq):
    """log P over the (tau, eps, alpha) grid; invalid combos (eps+alpha>1) get -inf."""
    alphas = ALPHA_GRID if with_eq else np.array([0.0])
    nA = len(game.actions); ne = game.nash(); un = np.full(nA, 1 / nA)
    T = np.full((len(TAU_GRID), len(EPS_GRID), len(alphas), nA), -np.inf)
    for i, t in enumerate(TAU_GRID):
        ch = ch_mix(game, t)
        for j, e in enumerate(EPS_GRID):
            for k, al in enumerate(alphas):
                if e + al <= 1 + 1e-9:
                    T[i, j, k] = np.log(np.maximum((1 - e - al) * ch + al * ne + e * un, 1e-300))
    return T, alphas

def elicited_level0(game_name, choices, mix_uniform=0.05):
    """S3 level 0 (fixed 2026-11-19 before fitting): the model's elicited 'non-strategic player' choices, smoothed with
    the game's tolerance kernel (beauty: +-3), mixed with 5% uniform so no action has zero level-0 probability."""
    g = make_game(game_name); a = g.actions; idx = {int(x): i for i, x in enumerate(a)}
    c = np.zeros(len(a))
    for x in choices: c[idx[int(x)]] += 1
    if g.kernel_sigma > 0: c = sum(c[i] * _smooth(a, a[i], g.kernel_sigma) for i in range(len(a)) if c[i] > 0)
    c = c / c.sum(); return (1 - mix_uniform) * c + mix_uniform / len(a)

def make_game_l0(name, l0):
    if name == "beauty": return Beauty(p=2/3, level0=l0)
    if name == "beauty_p05": return Beauty(p=0.5, level0=l0)
    if name == "req11_20": return Req1120(level0=l0)
    if name == "matrix3x3": return Matrix(level0=l0)
