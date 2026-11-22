import glob, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from fit import load
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8984", "axes.labelcolor": "#2b2b29", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.titleweight": "bold", "axes.titlesize": 11, "figure.dpi": 200, "savefig.bbox": "tight"})
COL = {"Claude-Haiku-4.5": "#2a78d6", "gpt-5-mini": "#eb6834", "gemini-2.5-flash": "#1baf7a", "llama3.2": "#eda100"}
NAME = {"Claude-Haiku-4.5": "Claude Haiku 4.5", "gpt-5-mini": "GPT-5-mini", "gemini-2.5-flash": "Gemini 2.5 Flash", "llama3.2": "Llama 3.2"}
MODELS = list(COL)
df = load(sorted(glob.glob('data/main_*.jsonl'))); df = df[df.valid]
OUT = "results/figs_extra/"

def light(hexc, f=0.55):
    c = np.array([int(hexc[i:i+2], 16) for i in (1, 3, 5)]) / 255; return tuple(c + (1 - c) * f)

# ---- F1: choice distributions, neutral framing, reasoning off vs on
for game, acts, xlab, marks in [("beauty", np.arange(0, 101), "guess (0-100)", {0: "Nash", 32: "L1", 20: "L2"}),
                                ("req11_20", np.arange(11, 21), "request ($)", {19: "L1", 18: "L2"}),
                                ("matrix3x3", np.arange(3), "action", {})]:
    fig, axs = plt.subplots(1, 4, figsize=(12, 2.6), sharey=True)
    for ax, m in zip(axs, MODELS):
        for cot, off in [(False, -0.2), (True, 0.2)]:
            d = df[(df.model == m) & (df.game == game) & (df.framing == "neutral") & (df.cot == cot)]
            if not len(d): continue
            if game == "beauty":
                edges = np.arange(0, 104, 4); cnt = np.histogram(d.choice, bins=edges)[0] / len(d); x = edges[:-1] + 2 + off * 4; w = 1.6
            else:
                cnt = np.array([(d.choice == a).sum() for a in acts]) / len(d); w = 0.38; x = acts + off
            ax.bar(x, cnt, width=w, color=COL[m] if cot else light(COL[m]), label="reasoning on" if cot else "reasoning off",
                   edgecolor="white", linewidth=0.5)
        ax.set_title(NAME[m], pad=16); ax.set_xlabel(xlab + (" (bins of 4)" if game == "beauty" else ""))
        if game == "matrix3x3": ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["A\n(Nash)", "B\n(dominated)", "C\n(L1)"])
        if game == "req11_20": ax.set_xticks(acts)
        for x0, lab in marks.items():
            ax.axvline(x0, color="#8a8984", lw=0.8, ls=":"); ax.text(x0, 1.02, lab, ha="center", va="bottom", fontsize=8, color="#52514e", transform=ax.get_xaxis_transform())
        ax.legend(frameon=False, fontsize=7, loc="upper right") if m == "Claude-Haiku-4.5" else None
    axs[0].set_ylabel("share of 100 calls")
    if game == "req11_20":
        for ax in axs: ax.plot(np.arange(15, 21), [.25, .25, .2, .15, .1, .05], color="#2b2b29", lw=1.2, marker="o", ms=3, label="Nash mix")
        axs[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.savefig(OUT + f"F1_choices_{game}.png"); plt.close(fig)

# ---- F2: tau / alpha per cell (CH + equilibrium model)
fits = pd.read_csv('results/main_fits.csv')
fig, axs = plt.subplots(1, 3, figsize=(12, 4.6), sharey=False)
for ax, game, title in zip(axs, ["beauty", "req11_20", "matrix3x3"], ["Beauty contest (p = 2/3)", "11-20 game", "3x3 game"]):
    f = fits[fits.game == game].copy(); f["mi"] = f.model.map({m: i for i, m in enumerate(MODELS)})
    f = f.sort_values(["mi", "framing", "cot"]).reset_index(drop=True); y = np.arange(len(f))[::-1]
    for yy, (_, r) in zip(y, f.iterrows()):
        ax.plot([r.eq_tau_lo, r.eq_tau_hi], [yy, yy], color=COL[r.model], lw=2, alpha=0.6)
        ax.plot(r.eq_tau, yy, "o", ms=7, color=COL[r.model] if r.cot else "white", mec=COL[r.model], mew=1.6)
        if r.eq_alpha >= 0.05: ax.text(6.25, yy, f"α={r.eq_alpha:.2f}", va="center", fontsize=7, color="#52514e")
    ax.set_yticks(y); ax.set_yticklabels([f"{NAME[r.model].split()[0]} · {'AI' if r.framing=='ai_competition' else 'neutral'} · {'on' if r.cot else 'off'}" for _, r in f.iterrows()], fontsize=7)
    ax.axvline(1.5, color="#8a8984", lw=0.8, ls="--"); ax.text(1.5, 1.01, "human τ≈1.5", transform=ax.get_xaxis_transform(), ha="center", fontsize=7, color="#52514e")
    ax.set_xlim(-0.1, 7); ax.set_xlabel("fitted τ (95% bootstrap interval)"); ax.set_title(title, pad=14)
fig.text(0.01, -0.02, "Filled = reasoning on, open = off. α = weight of the equilibrium type. τ at 5.9-6.0 = grid edge (not identified).", fontsize=8, color="#52514e")
fig.tight_layout(); fig.savefig(OUT + "F2_tau_forest.png"); plt.close(fig)

# ---- F3a: gain by round; F3b: frontier
tr = pd.read_csv('results/p2_traj.csv'); cells = pd.read_csv('results/p2_cells.csv')
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
SC = {"CH-online(0.0)": ("#2a78d6", "CH exploiter (ε=0)"), "FP": ("#eb6834", "fictitious play"), "RM": ("#1baf7a", "regret matching"), "CH-S2-online(0.0)": ("#6250d6", "CH exploiter, level 0 = 20")}
for k, (c, lab) in SC.items():
    t = tr[tr.strategy == k].groupby("round").gain.mean(); a1.plot(t.index, t.values, color=c, lw=2, label=lab)
orc = cells.groupby("opponent").oracle_gain.first().mean(); a1.axhline(orc, color="#2b2b29", lw=1, ls="--"); a1.text(20, orc + 0.1, "best response to true distribution", ha="right", fontsize=7, color="#52514e")
a1.axhline(0, color="#8a8984", lw=0.8); a1.set_xticks([1, 5, 10, 15, 20]); a1.set_xlabel("round"); a1.set_ylabel("gain over Nash play ($ per round)"); a1.set_title("Offline replay: learning curve (avg over 14 LLM opponents)"); a1.legend(frameon=False, fontsize=7)
fr = pd.read_csv('results/p2_frontier.csv'); e = fr[fr.eps.astype(str).str.match(r"^[0-9.]+$")].copy(); e["eps"] = e.eps.astype(float)
a2.plot(e.worst_gain_wrong_estimate, e.mean_gain_online, "-o", color="#2a78d6", ms=5, lw=2)
for _, r in e.iterrows():
    if r.eps in (0, 0.5, 1): a2.annotate(f"ε={r.eps:g}", (r.worst_gain_wrong_estimate, r.mean_gain_online), textcoords="offset points", xytext=(6, 4), fontsize=8)
a2.axvline(0, color="#8a8984", lw=0.8); a2.axhline(0, color="#8a8984", lw=0.8)
a2.set_xlabel("worst-case gain over Nash if the τ estimate is wrong ($/round)"); a2.set_ylabel("mean gain vs LLM opponents ($/round)"); a2.set_title("Safe-exploitation frontier (11-20 game)")
fig.tight_layout(); fig.savefig(OUT + "F3_phase2_offline.png"); plt.close(fig)

# ---- F4: live matches (15 matches x 10 rounds per model x strategy)
R = pd.read_csv('results/live_summary.csv')
lv = pd.DataFrame([json.loads(l) for l in open('data/live_1120.jsonl')]); lv["model"] = lv.model.str.replace("protected.", "")
LM = ["Claude-Haiku-4.5", "gpt-5-mini", "llama3.2"]; strat = ["ne", "exploit5", "exploit0"]; SL = {"ne": "Nash", "exploit5": "ε = 0.5", "exploit0": "ε = 0"}
fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, (col, lo, hi, title, yl) in zip(axs[:2], [("gain_vs_ne", "gain_lo", "gain_hi", "Exploiter's gain over Nash play", "$ per round (95% CI)"),
                                                ("llm_minus_exploiter", "dlo", "dhi", "Who wins? LLM payoff − exploiter payoff", "$ per round (95% CI)")]):
    for i, m in enumerate(LM):
        for j, s_ in enumerate(strat):
            r = R[(R.model == m) & (R.strategy == s_)].iloc[0]; x = j * 4 + i
            ax.bar(x, r[col], width=0.85, color=COL[m], edgecolor="white")
            ax.plot([x, x], [r[lo], r[hi]], color="#2b2b29", lw=1.2)
    ax.axhline(0, color="#8a8984", lw=0.8); ax.set_xticks([1, 5, 9]); ax.set_xticklabels([SL[s_] for s_ in strat]); ax.set_title(title); ax.set_ylabel(yl)
from matplotlib.patches import Patch
axs[0].legend(handles=[Patch(color=COL[m], label=NAME[m]) for m in LM], frameon=False, fontsize=7, loc="upper left")
ax = axs[2]; m = "gpt-5-mini"
for s_, c in zip(strat, ["#8a8984", "#6250d6", "#e34948"]):
    d = lv[(lv.model == m) & (lv.strategy == s_)].groupby("round").llm.mean(); ax.plot(d.index, d.values, color=c, lw=2, label=f"LLM vs {SL[s_]}")
d = lv[(lv.model == m) & (lv.strategy == "exploit0")].groupby("round").exploiter.mean(); ax.plot(d.index, d.values, color="#e34948", lw=1, ls=":", label="exploiter (ε=0) request")
ax.set_ylim(14, 20); ax.set_xticks([1, 4, 7, 10]); ax.set_xlabel("round"); ax.set_ylabel("mean request ($)"); ax.set_title("GPT-5-mini learns to undercut ε = 0"); ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(OUT + "F4_live.png"); plt.close(fig)
print("ok")
