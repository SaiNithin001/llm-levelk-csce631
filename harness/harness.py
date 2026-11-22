#!/usr/bin/env python3
"""Phase 1 sampling harness for CSCE 631 level-k project. Needs:  pip install openai   (Python 3.9+)
Follows the TAMU LLM API guide: base URL https://chat.tamu.ai/api, shared course key, CF_Authorization cookie, stream=True.
Environment variables:
  TAMU_API_KEY   the shared course key from the TAMU API guide
  TAMU_COOKIE    your CF_Authorization cookie (with or without the 'CF_Authorization=' prefix); expires ~24 h
  TAMU_API_BASE  optional, defaults to https://chat.tamu.ai/api

Commands:
  python3 harness.py check  [--models m1 m2 ...]            connectivity + list models + one test call each
  python3 harness.py run    --models m1 m2 ... --games beauty req11_20 --framings neutral --cot off on --n 20 --out pilot.jsonl
  python3 harness.py probe  --models m1 m2 ... --out nash_probe.jsonl      Nash-statement control (1 call/model/game x 3)
  python3 harness.py summary --out pilot.jsonl                              counts, parse-failure rate, tokens
'run' is resumable: re-running with the same --out only fills cells that are short of --n.
"""
import argparse, json, os, random, re, sys, time
from collections import Counter, defaultdict
from datetime import datetime, timezone
import prompts

TEMPERATURE = 1.0            # required by TAMU for Claude thinking models; used for every model for comparability
BASE_URL = os.environ.get("TAMU_API_BASE", "https://chat.tamu.ai/api")

def is_thinking(model):      # per the TAMU guide: Claude Sonnet / Opus run with extended thinking on the gateway
    return "Sonnet" in model or "Opus" in model

def max_tokens(model, cot):
    if is_thinking(model): return 16384          # must exceed the thinking budget; billed only for tokens produced
    if "gpt-5" in model: return 16384 if cot else 8192   # D4: main run used these caps for gpt-5-mini
    return 8192 if cot else 4096                 # pilot: gpt-5-mini / gemini-2.5-flash spent 512-2000 tokens on hidden
                                                 # reasoning and returned empty/truncated text; billed only for tokens produced

_client = None
_USE_STREAM_OPTIONS = True
def client():
    global _client
    if _client is None:
        import openai
        cookie = os.environ["TAMU_COOKIE"]
        if not cookie.startswith("CF_Authorization="): cookie = "CF_Authorization=" + cookie
        _client = openai.OpenAI(base_url=BASE_URL, api_key=os.environ["TAMU_API_KEY"], default_headers={"Cookie": cookie})
    return _client

class AuthError(Exception): pass

NONCE_MODE = "system"   # pilot showed the gateway returns cached replies to byte-identical requests
def make_nonce(rng=random.SystemRandom()):
    return "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(8))   # letters only: no numeric anchor

def build_messages(prompt, nonce):
    if NONCE_MODE == "system":
        return [{"role": "system", "content": f"Session code: {nonce}"}, {"role": "user", "content": prompt}]
    if NONCE_MODE == "user":
        return [{"role": "user", "content": f"Session code: {nonce}\n\n{prompt}"}]
    return [{"role": "user", "content": prompt}]

def call_model(model, prompt, max_toks, nonce=None):
    """Streaming call (the TAMU gateway requires stream=True). Returns (content, reasoning_content, served_model, usage)."""
    import openai
    global _USE_STREAM_OPTIONS
    kw = dict(model=model, messages=build_messages(prompt, nonce or make_nonce()), temperature=TEMPERATURE, max_tokens=max_toks, stream=True)
    try:
        try:
            stream = client().chat.completions.create(**kw, **({"stream_options": {"include_usage": True}} if _USE_STREAM_OPTIONS else {}))
        except openai.BadRequestError as e:
            if _USE_STREAM_OPTIONS and "stream_options" in str(e):   # gateway rejects it: continue without token usage
                _USE_STREAM_OPTIONS = False
                stream = client().chat.completions.create(**kw)
            else:
                raise
    except (openai.AuthenticationError, openai.PermissionDeniedError) as e:
        raise AuthError(str(e))
    text, reasoning, served, usage = "", "", None, {}
    for ch in stream:
        served = getattr(ch, "model", None) or served
        if getattr(ch, "usage", None):
            u = ch.usage; usage = {"prompt_tokens": getattr(u, "prompt_tokens", 0), "completion_tokens": getattr(u, "completion_tokens", 0),
                                   "total_tokens": getattr(u, "total_tokens", 0)}
        if ch.choices:
            d = ch.choices[0].delta
            if getattr(d, "content", None): text += d.content
            rc = getattr(d, "reasoning_content", None) or (getattr(d, "model_extra", None) or {}).get("reasoning_content")
            if rc: reasoning += rc
    return text, reasoning, served or model, usage

def call_with_retry(model, prompt, max_toks, tries=4, nonce=None):
    import openai
    err = None
    for i in range(tries):
        try:
            return call_model(model, prompt, max_toks, nonce), None
        except AuthError as e:
            print("\n401/403 from the gateway: your CF_Authorization cookie has probably expired (~24 h).\n"
                  "Log into https://chat.tamu.ai again, copy a fresh cookie into TAMU_COOKIE, and re-run the same command (it resumes).")
            sys.exit(2)
        except openai.NotFoundError as e:
            return None, f"model not found: {e}"[:300]
        except openai.BadRequestError as e:
            return None, f"bad request: {e}"[:300]
        except Exception as e:                       # 429, connection errors, gateway hiccups: back off and retry
            err = f"{type(e).__name__}: {e}"[:300]
        time.sleep(2 ** i * 2)
    return None, err

# ------------------------------------------------------------------ parsing (strict, pre-registered)
ANS = re.compile(r"ANSWER\s*:\s*\**\s*\$?\s*([A-Ca-c]|\d{1,3})\b")
def parse(game, text):
    """Use the LAST 'ANSWER:' line. Valid only if inside the action set. No fallback guessing."""
    m = ANS.findall(text or "")
    if not m: return None, "no ANSWER line"
    v = m[-1].strip()
    if game == "matrix3x3":
        v = v.upper()
        return (v, None) if v in "ABC" else (None, f"bad action {v}")
    if not v.isdigit(): return None, f"non-integer {v}"
    x = int(v); lo, hi = (11, 20) if game == "req11_20" else (0, 100)
    return (x, None) if lo <= x <= hi else (None, f"out of range {x}")

MATRIX_IDX = {"A": 0, "B": 1, "C": 2}   # fit_cells.py uses 0,1,2

# ------------------------------------------------------------------ commands
def cmd_check(a):
    print("API base:", BASE_URL)
    try:
        ids = sorted(m.id for m in client().models.list())
        print(f"models.list() returned {len(ids)} models:"); [print("  ", i) for i in ids]
    except Exception as e:
        print("models.list() failed:", type(e).__name__, e)
    for m in a.models or []:
        res, err = call_with_retry(m, "Reply with exactly one line: ANSWER: 7", max_tokens(m, False), tries=2)
        if res: print(f"[{m}] content={res[0][:80]!r} | reasoning chars={len(res[1])} | served as {res[2]} | usage {res[3]}")
        else:   print(f"[{m}] ERROR {err}")

def cmd_dupcheck(a):
    """Send the same CoT prompt 5 times per nonce mode and count distinct replies (cache test)."""
    global NONCE_MODE
    p = prompts.build("beauty", "neutral", True)
    for mode in ["none", "system", "user"]:
        NONCE_MODE = mode; outs = []
        for _ in range(a.n):
            res, err = call_with_retry(a.models[0], p, max_tokens(a.models[0], True))
            outs.append(res[0] if res else f"ERR {err}")
        print(f"nonce={mode:6s}: {len(set(outs))} distinct replies out of {len(outs)}; answers {[parse('beauty', o)[0] for o in outs]}")

def _existing(path):
    c = Counter()
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l); c[(r["model"], r["game"], r["framing"], r["cot"])] += 1
    return c

def cmd_run(a):
    have = _existing(a.out)
    cells = [(m, g, f, c == "on") for m in a.models for g in a.games for f in a.framings for c in a.cot]
    jobs = []
    for cell in cells:
        jobs += [cell] * max(0, a.n - have[cell])
    random.Random(a.shuffle_seed).shuffle(jobs)          # interleave cells so drift/outages don't hit one cell
    print(f"{len(jobs)} calls to make ({sum(have.values())} already logged in {a.out})")
    tok = 0
    with open(a.out, "a") as f:
        for i, (m, g, fr, cot) in enumerate(jobs, 1):
            prompt = prompts.build(g, fr, cot); call_id = random.randrange(2**31)
            nonce = make_nonce()
            t0 = time.time(); res, err = call_with_retry(m, prompt, max_tokens(m, cot), nonce=nonce)
            text, reasoning, served, usage = res if res else ("", "", None, {})
            val, perr = parse(g, text) if res else (None, err)
            choice = MATRIX_IDX[val] if (g == "matrix3x3" and val is not None) else val
            rec = dict(model=m, served_model=served, game=g, framing=fr, cot=cot, choice=choice, valid=val is not None,
                       parse_error=perr, raw=text, reasoning=reasoning, thinking_model=is_thinking(m),
                       prompt_id=f"{g}|{fr}|{'cot' if cot else 'nocot'}", call_id=call_id, nonce=nonce, nonce_mode=NONCE_MODE, max_tokens=max_tokens(m, cot),
                       temperature=TEMPERATURE, usage=usage, latency_s=round(time.time() - t0, 2),
                       ts=datetime.now(timezone.utc).isoformat())
            f.write(json.dumps(rec) + "\n"); f.flush()
            tok += (usage or {}).get("total_tokens", 0) or 0
            if i % 10 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done, tokens so far {tok}", flush=True)
            time.sleep(a.sleep)

def cmd_probe(a):
    with open(a.out, "a") as f:
        for m in a.models:
            for g in (a.games if a.games != ["beauty", "req11_20"] else ["beauty", "req11_20", "matrix3x3"]):
                for rep in range(3):
                    res, err = call_with_retry(m, prompts.nash_probe(g), max_tokens(m, True))
                    text = res[0] if res else ""
                    mm = re.findall(r"NASH\s*:\s*(.+)", text)
                    f.write(json.dumps(dict(model=m, served_model=res[1] if res else None, game=g, rep=rep,
                                            nash_answer=mm[-1].strip() if mm else None, raw=text, error=err,
                                            ts=datetime.now(timezone.utc).isoformat())) + "\n")
                    print(m, g, rep, "->", mm[-1][:80] if mm else err)

def cmd_summary(a):
    rows = [json.loads(l) for l in open(a.out)]
    by = defaultdict(list)
    for r in rows: by[(r["model"], r["game"], r["framing"], "cot" if r["cot"] else "nocot")].append(r)
    print(f"{'cell':60s} n  valid  distinct  top choices                     tokens")
    for k, rs in sorted(by.items()):
        v = [r["choice"] for r in rs if r["valid"]]
        t = sum((r.get("usage") or {}).get("total_tokens", 0) or 0 for r in rs)
        d = len(set(r["raw"] for r in rs))
        print(f"{'|'.join(map(str,k)):60s} {len(rs):3d} {len(v):4d}  {d:6d}    {str(Counter(v).most_common(4)):32s} {t}")
    errs = Counter(r["parse_error"] for r in rows if not r["valid"])
    if errs: print("parse/API errors:", errs.most_common(8))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["check", "run", "probe", "summary", "dupcheck"])
    ap.add_argument("--nonce", choices=["system", "user", "none"], default="system")
    ap.add_argument("--models", nargs="*"); ap.add_argument("--games", nargs="*", default=["beauty", "req11_20"])
    ap.add_argument("--framings", nargs="*", default=["neutral"]); ap.add_argument("--cot", nargs="*", default=["off"])
    ap.add_argument("--n", type=int, default=20); ap.add_argument("--out", default="pilot.jsonl")
    ap.add_argument("--sleep", type=float, default=1.0); ap.add_argument("--shuffle_seed", type=int, default=631)
    a = ap.parse_args(); NONCE_MODE = a.nonce
    {"dupcheck": cmd_dupcheck, "check": cmd_check, "run": cmd_run, "probe": cmd_probe, "summary": cmd_summary}[a.cmd](a)
