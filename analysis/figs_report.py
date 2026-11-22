import glob, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from fit import load
INK, SUB, RULE = "#1E1B1A", "#4A4542", "#9A938C"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": RULE, "axes.labelcolor": INK, "xtick.color": SUB, "ytick.color": SUB, "axes.titleweight": "bold",
    "axes.titlesize": 9.5, "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "savefig.bbox": "tight"})
COL = {"Claude-Haiku-4.5": "#2a78d6", "gpt-5-mini": "#eb6834", "gemini-2.5-flash": "#1baf7a", "llama3.2": "#c98500"}
NAME = {"Claude-Haiku-4.5": "Claude Haiku 4.5", "gpt-5-mini": "GPT-5-mini", "gemini-2.5-flash": "Gemini 2.5 Flash", "llama3.2": "Llama 3.2"}
M = list(COL); OUT = "report/figs/"
def light(h, f=0.55):
    c = np.array([int(h[i:i+2], 16) for i in (1, 3, 5)]) / 255; return tuple(c + (1 - c) * f)
df = load(sorted(glob.glob('data/main_*.jsonl'))); df = df[df.valid]
c30 = load(['data/c30.jsonl']); c30 = c30[c30.valid]

def hist(ax, d, game, color, off, label):
    if game.startswith("beauty"):
        e = np.arange(0, 104, 4); c = np.histogram(d.choice.astype(int), bins=e)[0] / max(len(d), 1); x = e[:-1] + 2 + off * 4; w = 1.6
    else:
        a = np.arange(11, 21); c = np.array([(d.choice == v).sum() for v in a]) / max(len(d), 1); x = a + off; w = 0.4
    ax.bar(x, c, width=w, color=color, label=label)

# Fig 1: choices, neutral framing
fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.3), sharey="row")
for r, game in enumerate(["beauty", "req11_20"]):
    for ax, m in zip(axs[r], M):
        for cot, off in [(False, -0.2), (True, 0.2)]:
            d = df[(df.model == m) & (df.game == game) & (df.framing == "neutral") & (df.cot == cot)]
            if len(d): hist(ax, d, game, COL[m] if cot else light(COL[m]), off, "reasoning on" if cot else "reasoning off")
        if game == "beauty":
            for x0, lab in [(0, "NE"), (20, "L2"), (32, "L1")]:
                ax.axvline(x0, color=RULE, lw=0.7, ls=":"); ax.text(x0, 1.0, lab, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.5, color=SUB)
            ax.set_xlabel("guess")
        else:
            ax.plot(np.arange(15, 21), [.25, .25, .2, .15, .1, .05], color=INK, lw=1, marker="o", ms=2, label="NE mix"); ax.set_xticks([11, 13, 15, 17, 19]); ax.set_xlabel("request ($)")
        if r == 0: ax.set_title(NAME[m], pad=10)
        ax.set_ylim(0, 1.05)
    axs[r][0].set_ylabel("beauty contest\nshare" if r == 0 else "11–20 game\nshare")
axs[0][0].legend(frameon=False, loc="upper right"); axs[1][0].legend(frameon=False, loc="upper left", handles=axs[1][0].get_legend_handles_labels()[0][-1:])
fig.tight_layout(h_pad=1.0); fig.savefig(OUT + "fig1_choices.pdf"); plt.close(fig)

# Fig 3: standard (Nash 0) vs shifted (Nash 90) beauty contest, neutral framing
fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.0), sharey=True)
for r, cot in enumerate([False, True]):
    for ax, m in zip(axs[r], M):
        s = df[(df.model == m) & (df.game == "beauty") & (df.framing == "neutral") & (df.cot == cot)]
        t = c30[(c30.model == m) & (c30.cot == cot)]
        if len(s): hist(ax, s, "beauty", light(COL[m]), -0.2, "target ⅔·avg (NE 0)")
        if len(t): hist(ax, t, "beauty", COL[m], 0.2, "target ⅔·avg + 30 (NE 90)")
        if not len(s) and not len(t): ax.text(50, 0.5, "not run", ha="center", color=SUB, fontsize=8)
        for x0, lab in [(0, "0"), (30, "30"), (65, "L1"), (90, "NE")]:
            ax.axvline(x0, color=RULE, lw=0.7, ls=":")
            if r == 0: ax.text(x0, 1.0, lab, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.5, color=SUB)
        if r == 0: ax.set_title(NAME[m], pad=10)
        if r == 1: ax.set_xlabel("guess")
        ax.set_ylim(0, 1.05)
    axs[r][0].set_ylabel(f"reasoning {'on' if cot else 'off'}\nshare")
h, l = axs[0][0].get_legend_handles_labels()
fig.tight_layout(h_pad=1.0, rect=(0, 0.06, 1, 1)); fig.legend(h, ["standard: target ⅔·avg (NE 0)", "shifted: target ⅔·avg + 30 (NE 90)"], loc="lower center", ncol=2, frameon=False, fontsize=7.5)
fig.savefig(OUT + "fig3_c30.pdf"); plt.close(fig)

# Fig 5: live, 4 strategies + adaptive epsilon trajectory
R = pd.read_csv('results/live_summary.csv')
lv = pd.DataFrame([json.loads(l) for l in open('data/live_1120.jsonl')]); lv["model"] = lv.model.str.replace("protected.", "")
LM = ["Claude-Haiku-4.5", "gpt-5-mini", "llama3.2"]; S = ["ne", "exploit5", "exploit0", "adaptive"]; SL = {"ne": "NE", "exploit5": "ε=0.5", "exploit0": "ε=0", "adaptive": "adaptive ε"}
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3), gridspec_kw={"width_ratios": [1.15, 1.15, 1]})
for ax, (col, lo, hi, title) in zip(axs[:2], [("gain_vs_ne", "gain_lo", "gain_hi", "(a) exploiter gain over NE play"), ("llm_minus_exploiter", "dlo", "dhi", "(b) LLM payoff − exploiter payoff")]):
    for i, m in enumerate(LM):
        for j, s in enumerate(S):
            r = R[(R.model == m) & (R.strategy == s)].iloc[0]; x = j * 4 + i
            ax.bar(x, r[col], width=0.85, color=COL[m]); ax.plot([x, x], [r[lo], r[hi]], color=INK, lw=0.9)
    ax.axhline(0, color=RULE, lw=0.7); ax.set_xticks([1, 5, 9, 13]); ax.set_xticklabels([SL[s] for s in S], fontsize=7.5); ax.set_title(title); ax.set_ylabel("$ per round (95% CI)")
axs[0].legend(handles=[Patch(color=COL[m], label=NAME[m]) for m in LM], frameon=False, loc="upper left", fontsize=6.5)
ax = axs[2]
for m in LM:
    t = lv[(lv.model == m) & (lv.strategy == "adaptive")].groupby("round").eps.mean(); ax.plot(t.index, t.values, color=COL[m], lw=1.6, label=NAME[m])
ax.set_ylim(0, 1); ax.set_xticks([1, 4, 7, 10]); ax.set_xlabel("round"); ax.set_ylabel("mean ε (P(adaptive))"); ax.set_title("(c) adaptive exploiter’s ε"); ax.legend(frameon=False, fontsize=6.5)
fig.tight_layout(); fig.savefig(OUT + "fig5_live.pdf"); plt.close(fig)

# Fig 4: offline learning curve
tr = pd.read_csv('results/p2_traj.csv'); cells = pd.read_csv('results/p2_cells.csv'); fr = pd.read_csv('results/p2_frontier.csv')
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.35))
for k, c, lab in [("CH-online(0.0)", "#2a78d6", "CH exploiter (ε=0)"), ("FP", "#eb6834", "fictitious play"), ("RM", "#1baf7a", "regret matching")]:
    t = tr[tr.strategy == k].groupby("round").gain.mean(); a1.plot(t.index, t.values, color=c, lw=1.6, label=lab)
orc = cells.groupby("opponent").oracle_gain.first().mean(); a1.axhline(orc, color=INK, lw=0.8, ls="--"); a1.text(20, orc + 0.15, "BR to true distribution", ha="right", fontsize=6.5, color=SUB)
a1.axhline(0, color=RULE, lw=0.7); a1.set_xticks([1, 5, 10, 15, 20]); a1.set_xlabel("round"); a1.set_ylabel("gain over NE ($/round)"); a1.set_title("(a) learning curve, 14 opponents"); a1.legend(frameon=False, loc="lower right")
e = fr[fr.eps.astype(str).str.match(r"^[0-9.]+$")].copy(); e["eps"] = e.eps.astype(float)
a2.plot(e.eps, e.mean_gain_online, "-o", color="#2a78d6", ms=3, lw=1.4, label="mean gain, online posterior")
a2.plot(e.eps, e.worst_gain_wrong_estimate, "-s", color="#c0392b", ms=3, lw=1.4, label="worst case, fixed wrong τ′")
a2.axhline(0, color=RULE, lw=0.7); a2.set_xlabel("ε (weight on NE)"); a2.set_ylabel("gain over NE ($/round)"); a2.set_title("(b) safety trade-off"); a2.legend(frameon=False, fontsize=6.5)
fig.tight_layout(); fig.savefig(OUT + "fig4_offline.pdf"); plt.close(fig)
print("ok")

# Fig 2: tau forest, all conditions
fits = pd.read_csv('results/main_fits.csv')
fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.7))
for ax, game, title in zip(axs, ["beauty", "req11_20", "matrix3x3"], ["Beauty contest (p = 2/3)", "11–20 game", "3×3 game"]):
    f = fits[fits.game == game].copy(); f["mi"] = f.model.map({m: i for i, m in enumerate(M)})
    f = f.sort_values(["mi", "framing", "cot"], ascending=[True, False, True]).reset_index(drop=True); y = np.arange(len(f))[::-1]
    for yy, (_, r) in zip(y, f.iterrows()):
        ax.plot([r.eq_tau_lo, r.eq_tau_hi], [yy, yy], color=COL[r.model], lw=1.6, alpha=0.6)
        ax.plot(r.eq_tau, yy, "o", ms=4.5, color=COL[r.model] if r.cot else "white", mec=COL[r.model], mew=1.2)
        if r.eq_alpha >= 0.05: ax.text(6.2, yy, f"α {r.eq_alpha:.2f}", va="center", fontsize=5.5, color=SUB)
    ax.set_yticks(y); ax.set_yticklabels([f"{NAME[r.model].split()[0]} · {'AI' if r.framing=='ai_competition' else 'neu.'} · {'on' if r.cot else 'off'}" for _, r in f.iterrows()], fontsize=6.5)
    ax.axvline(1.5, color=RULE, lw=0.7, ls="--"); ax.set_xlim(-0.1, 7.3); ax.set_xticks([0, 2, 4, 6]); ax.set_xlabel("fitted τ (95% CI)"); ax.set_title(title)
fig.tight_layout(w_pad=2.2); fig.savefig(OUT + "fig2_tau.pdf"); plt.close(fig)
print("fig2 ok")
