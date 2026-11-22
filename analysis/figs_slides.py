import glob, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from fit import load
INK, SUB, RULE = "#1E1B1A", "#4A4542", "#9A938C"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": RULE, "axes.labelcolor": INK, "xtick.color": SUB, "ytick.color": SUB, "axes.titleweight": "bold",
    "axes.titlesize": 17, "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13,
    "figure.dpi": 120, "savefig.bbox": "tight", "savefig.facecolor": "#FAF8F5", "figure.facecolor": "#FAF8F5", "axes.facecolor": "#FAF8F5"})
COL = {"Claude-Haiku-4.5": "#2a78d6", "gpt-5-mini": "#eb6834", "gemini-2.5-flash": "#1baf7a", "llama3.2": "#c98500"}
NAME = {"Claude-Haiku-4.5": "Claude Haiku 4.5", "gpt-5-mini": "GPT-5-mini", "gemini-2.5-flash": "Gemini 2.5 Flash", "llama3.2": "Llama 3.2"}
M = list(COL); OUT = "results/figs_slides/"
def light(h, f=0.55):
    c = np.array([int(h[i:i+2], 16) for i in (1, 3, 5)]) / 255; return tuple(c + (1 - c) * f)
df = load(sorted(glob.glob('data/main_*.jsonl'))); df = df[df.valid]

# choices: 2 rows (beauty, 11-20) x 4 models, neutral framing
fig, axs = plt.subplots(2, 4, figsize=(16, 6.6), sharey="row")
for r, (game, xl) in enumerate([("beauty", "guess (bins of 4)"), ("req11_20", "request ($)")]):
    for ax, m in zip(axs[r], M):
        for cot, off in [(False, -0.2), (True, 0.2)]:
            d = df[(df.model == m) & (df.game == game) & (df.framing == "neutral") & (df.cot == cot)]
            if not len(d): continue
            if game == "beauty":
                e = np.arange(0, 104, 4); c = np.histogram(d.choice, bins=e)[0] / len(d); x = e[:-1] + 2 + off * 4; w = 1.6
            else:
                a = np.arange(11, 21); c = np.array([(d.choice == v).sum() for v in a]) / len(d); x = a + off; w = 0.4
            ax.bar(x, c, width=w, color=COL[m] if cot else light(COL[m]), label="reasoning on" if cot else "reasoning off")
        if game == "beauty":
            for x0, lab in [(0, "Nash"), (20, "L2"), (32, "L1")]:
                ax.axvline(x0, color=RULE, lw=1, ls=":"); ax.text(x0, 1.0, lab, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=12, color=SUB)
        else:
            ax.plot(np.arange(15, 21), [.25, .25, .2, .15, .1, .05], color=INK, lw=1.5, marker="o", ms=4, label="Nash mix")
            ax.set_xticks([11, 13, 15, 17, 19])
        ax.set_xlabel(xl)
        if r == 0: ax.set_title(NAME[m], pad=22)
        ax.set_ylim(0, 1.05)
    axs[r][0].set_ylabel("Beauty contest\nshare of calls" if r == 0 else "11–20 game\nshare of calls")
axs[0][0].legend(frameon=False, loc="upper right", fontsize=12)
h = [l for l in axs[1][2].get_lines()]; axs[1][2].legend(h[:1], ["Nash equilibrium mix"], frameon=False, loc="upper center", fontsize=12)
fig.tight_layout(h_pad=1.5); fig.savefig(OUT + "choices.png"); plt.close(fig)

# offline learning curve
tr = pd.read_csv('results/p2_traj.csv'); cells = pd.read_csv('results/p2_cells.csv')
fig, ax = plt.subplots(figsize=(9.5, 6.2))
for k, c, lab in [("CH-online(0.0)", "#2a78d6", "CH exploiter (model-based)"), ("FP", "#eb6834", "Fictitious play (model-free)"), ("RM", "#1baf7a", "Regret matching (model-free)")]:
    t = tr[tr.strategy == k].groupby("round").gain.mean(); ax.plot(t.index, t.values, color=c, lw=3, label=lab)
orc = cells.groupby("opponent").oracle_gain.first().mean()
ax.axhline(orc, color=INK, lw=1.2, ls="--"); ax.text(20, orc + 0.15, "best response to the true distribution", ha="right", fontsize=12, color=SUB)
ax.axhline(0, color=RULE, lw=1); ax.text(20, 0.15, "Nash play", ha="right", fontsize=12, color=SUB)
ax.set_xticks([1, 5, 10, 15, 20]); ax.set_xlabel("round"); ax.set_ylabel("gain over Nash play ($ per round)")
ax.legend(frameon=False, loc="lower right"); fig.savefig(OUT + "offline.png"); plt.close(fig)

# live
R = pd.read_csv('results/live_summary.csv')
lv = pd.DataFrame([json.loads(l) for l in open('data/live_1120.jsonl')]); lv["model"] = lv.model.str.replace("protected.", "")
LM = ["Claude-Haiku-4.5", "gpt-5-mini", "llama3.2"]; S = ["ne", "exploit5", "exploit0"]; SL = {"ne": "Nash", "exploit5": "ε = 0.5", "exploit0": "ε = 0"}
fig, axs = plt.subplots(1, 3, figsize=(16, 5.6))
from matplotlib.patches import Patch
for ax, (col, lo, hi, title) in zip(axs[:2], [("gain_vs_ne", "gain_lo", "gain_hi", "Exploiter’s gain over Nash play"), ("llm_minus_exploiter", "dlo", "dhi", "LLM payoff − exploiter payoff")]):
    for i, m in enumerate(LM):
        for j, s in enumerate(S):
            r = R[(R.model == m) & (R.strategy == s)].iloc[0]; x = j * 4 + i
            ax.bar(x, r[col], width=0.85, color=COL[m]); ax.plot([x, x], [r[lo], r[hi]], color=INK, lw=1.6)
    ax.axhline(0, color=RULE, lw=1); ax.set_xticks([1, 5, 9]); ax.set_xticklabels([SL[s] for s in S]); ax.set_title(title); ax.set_ylabel("$ per round (95% CI)")
axs[0].legend(handles=[Patch(color=COL[m], label=NAME[m]) for m in LM], frameon=False, loc="upper left", fontsize=12)
ax = axs[2]
for s, c in zip(S, [RULE, "#6250d6", "#c0392b"]):
    d = lv[(lv.model == "gpt-5-mini") & (lv.strategy == s)].groupby("round").llm.mean(); ax.plot(d.index, d.values, color=c, lw=3, label=f"GPT-5-mini vs {SL[s]}")
d = lv[(lv.model == "gpt-5-mini") & (lv.strategy == "exploit0")].groupby("round").exploiter.mean(); ax.plot(d.index, d.values, color="#c0392b", lw=1.5, ls=":", label="exploiter (ε = 0)")
ax.set_ylim(14.5, 19.5); ax.set_xticks([1, 4, 7, 10]); ax.set_xlabel("round"); ax.set_ylabel("mean request ($)"); ax.set_title("GPT-5-mini learns to undercut"); ax.legend(frameon=False, fontsize=11, loc="upper right")
fig.tight_layout(); fig.savefig(OUT + "live.png"); plt.close(fig)
print("ok")
