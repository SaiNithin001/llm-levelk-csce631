#!/usr/bin/env python3
"""Phase 2 live check: repeated 11-20 game, LLM vs a scripted exploiter, with feedback.
Each round is one stateless call; the prompt contains the full history (both requests and both payoffs).
The exploiter keeps a Bayesian posterior over the CH+equilibrium model (exploiter_table.npz, precomputed by the
analysis code) updated on the LLM's requests and plays (1-e)*BestResponse + e*Equilibrium, sampled each round.
Strategies: ne (e=1), exploit0 (e=0), exploit5 (e=0.5).  Needs numpy (+ the harness's openai setup).
Usage:
  python3 live_1120.py --models "protected.Claude-Haiku-4.5" "protected.gpt-5-mini" --matches 5 --rounds 10 --out live_1120.jsonl
Resumable: completed (model, strategy, match) triples in --out are skipped."""
import argparse, json, os, random, zlib
from datetime import datetime, timezone
import numpy as np
import harness

ACTS = list(range(11, 21))
U = np.array([[r + 20.0 * (r == s - 1) for s in ACTS] for r in ACTS])
NE = np.array([0, 0, 0, 0, .25, .25, .20, .15, .10, .05])          # over 11..20
TAB = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "exploiter_table.npz"))
LOGP, P = TAB["logp"], TAB["p"]
STRATS = {"ne": 1.0, "exploit0": 0.0, "exploit5": 0.5, "adaptive": "adaptive"}
# Adaptive-epsilon exploiter (added 2026-11-22, after the main results; rule fixed before its data):
# two hypotheses about the LLM. STATIC: it draws from the CH model (posterior predictive w @ P).
# ADAPTIVE: it best-responds to the exploiter's previous request r, i.e. plays r-1 (20 if r = 11)
# with prob. 0.7, else uniform. eps_t = P(ADAPTIVE | history), prior 0.1; round 1 has no evidence.
PRIOR_ADAPT, HIT = 0.1, 0.7
def adapt_lik(y, prev_x):
    target = 20 if prev_x == 11 else prev_x - 1
    return HIT * (y == target) + (1 - HIT) / 10

def payoff(a, b): return a + (20 if a == b - 1 else 0)

def prompt(history, rounds):
    lines = [f"You are a participant in a decision-making study. You will play the same game with the same other "
             f"participant for {rounds} rounds.\n",
             "Each round, you and the other participant each request an amount of money: a whole number of dollars "
             "from 11 to 20 (inclusive). Each of you receives the amount you requested. In addition, if your request "
             "is exactly one dollar less than the other's request, you receive an extra $20.\n"]
    if history:
        lines.append("Results so far:")
        for i, (me, other) in enumerate(history, 1):
            lines.append(f"- Round {i}: you requested ${me}, the other requested ${other}; you earned ${payoff(me, other)}, they earned ${payoff(other, me)}.")
        lines.append("")
    lines.append(f"This is round {len(history) + 1} of {rounds}. How much do you request?\n\n"
                 "Do not explain your reasoning. Reply with exactly one line of the form\nANSWER: <a whole number from 11 to 20>")
    return "\n".join(lines)

def exploiter_action(logpost, e, rng):
    w = np.exp(logpost - logpost.max()); w /= w.sum()
    q = w @ P; br = int(np.argmax(U @ q))
    mix = (1 - e) * np.eye(10)[br] + e * NE
    return ACTS[rng.choice(10, p=mix)], ACTS[br]

def done(path):
    s = set()
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l)
            if r.get("round") == r.get("rounds"): s.add((r["model"], r["strategy"], r["match"]))
    return s

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--matches", type=int, default=5); ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--strategies", nargs="+", default=list(STRATS)); ap.add_argument("--out", default="live_1120.jsonl")
    a = ap.parse_args(); fin = done(a.out)
    with open(a.out, "a") as f:
        for m in a.models:
            for s in a.strategies:
                for k in range(a.matches):
                    if (m, s, k) in fin: continue
                    rng = np.random.default_rng(zlib.crc32(f"{m}|{s}|{k}".encode())); logpost = np.zeros(LOGP.shape[0]); hist = []
                    log_odds = np.log(PRIOR_ADAPT / (1 - PRIOR_ADAPT))
                    for t in range(1, a.rounds + 1):
                        eps = float(1 / (1 + np.exp(-log_odds))) if s == "adaptive" else STRATS[s]
                        x, br = exploiter_action(logpost, eps, rng)
                        res, err = harness.call_with_retry(m, prompt(hist, a.rounds), harness.max_tokens(m, False))
                        text = res[0] if res else ""
                        y, perr = harness.parse("req11_20", text) if res else (None, err)
                        if y is None: y = 20; invalid = True          # a failed reply counts as 20 (logged as invalid)
                        else: invalid = False
                        if hist:                                  # evidence for ADAPTIVE vs STATIC
                            w = np.exp(logpost - logpost.max()); w /= w.sum()
                            log_odds += np.log(adapt_lik(y, hist[-1][1])) - np.log(max(float((w @ P)[y - 11]), 1e-12))
                            log_odds = float(np.clip(log_odds, -20, 20))
                        logpost += LOGP[:, y - 11]; hist.append((y, x))
                        f.write(json.dumps(dict(model=m, strategy=s, eps=eps, match=k, round=t, rounds=a.rounds,
                                                llm=y, exploiter=x, exploiter_br=br, llm_payoff=payoff(y, x),
                                                exploiter_payoff=payoff(x, y), ne_payoff_vs_llm=float(NE @ U[:, y - 11]),
                                                invalid=invalid, parse_error=perr, raw=text, usage=res[3] if res else {},
                                                ts=datetime.now(timezone.utc).isoformat())) + "\n"); f.flush()
                    print(f"{m} {s} match {k}: llm {[h[0] for h in hist]}  exploiter {[h[1] for h in hist]}", flush=True)

if __name__ == "__main__":
    main()
