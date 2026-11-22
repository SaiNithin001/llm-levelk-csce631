#!/usr/bin/env python3
"""Pre-registered sensitivity S3: elicit each model's belief about a NON-strategic (level-0) player.
Same game text as the main run (neutral framing), but the question asks what a player who does not reason about
the others would choose. Answers are parsed exactly like the main run. Resumable.
Usage:
  python3 elicit_l0.py --models "protected.Claude-Haiku-4.5" "protected.gpt-5-mini" "protected.gemini-2.5-flash" "protected.llama3.2" --n 30 --out elicit_l0.jsonl"""
import argparse, json, os, random
from collections import Counter
from datetime import datetime, timezone
import harness, prompts

QUESTION = {"beauty": "Which number do you choose?", "req11_20": "How much do you request?", "matrix3x3": "Which action do you choose?"}

def build(game):
    body = {"beauty": prompts.beauty("neutral"), "req11_20": prompts.req1120("neutral"), "matrix3x3": prompts.matrix3x3("neutral")}[game]
    body = body.replace(QUESTION[game], "").rstrip()
    return (body + "\n\nDo not choose for yourself. Instead, imagine another player who does not think about what the "
            "other players will do and chooses without any strategic reasoning. What would such a player most likely choose?"
            f"\n\nDo not explain your reasoning. Reply with exactly one line of the form\nANSWER: <{prompts.ANSWER_SPEC[game]}>")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--games", nargs="+", default=["beauty", "req11_20", "matrix3x3"])
    ap.add_argument("--n", type=int, default=30); ap.add_argument("--out", default="elicit_l0.jsonl")
    a = ap.parse_args()
    have = Counter()
    if os.path.exists(a.out):
        for l in open(a.out): r = json.loads(l); have[(r["model"], r["game"])] += 1
    jobs = [(m, g) for m in a.models for g in a.games for _ in range(max(0, a.n - have[(m, g)]))]
    random.Random(631).shuffle(jobs); print(f"{len(jobs)} calls to make")
    with open(a.out, "a") as f:
        for i, (m, g) in enumerate(jobs, 1):
            res, err = harness.call_with_retry(m, build(g), harness.max_tokens(m, False))
            text = res[0] if res else ""
            val, perr = harness.parse(g, text) if res else (None, err)
            choice = harness.MATRIX_IDX[val] if (g == "matrix3x3" and val is not None) else val
            f.write(json.dumps(dict(model=m, game=g, task="level0_elicitation", choice=choice, valid=val is not None,
                                    parse_error=perr, raw=text, usage=res[3] if res else {},
                                    ts=datetime.now(timezone.utc).isoformat())) + "\n"); f.flush()
            if i % 20 == 0 or i == len(jobs): print(f"  {i}/{len(jobs)}", flush=True)

if __name__ == "__main__":
    main()
