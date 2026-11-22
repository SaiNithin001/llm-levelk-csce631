# How Deep Do LLMs Think? — code and data (CSCE 631-D, Fall 2026, Sai Maddi)
Version used for the final report: [ADD COMMIT HASH OR RELEASE TAG] (Dec 2026).

## Contents
- `harness/`: sampling code for the TAMU AI API gateway (`harness.py`, `prompts.py`), the live repeated-game exploiter (`live_1120.py`, with its precomputed posterior table `exploiter_table.npz`), and the level-0 elicitation (`elicit_l0.py`).
- `analysis/`: the CH model (`levelk.py`), fitting and bootstrap (`fit.py`), population fit (`pooled.py`), 67 pre-registered comparisons (`tests.py`), offline exploitation (`phase2.py`), and summaries and figures (`run_*.py`, `figs_report.py`).
- `data/`: one JSON line per API call (prompt id, raw reply, served model, token usage, session code, UTC timestamp).
  - main run (`main_*.jsonl`); p = 1/2 control (`main_p05.jsonl`); shifted contest (`c30.jsonl`)
  - Nash probes (`nash_probe*.jsonl`); level-0 elicitation (`elicit_l0.jsonl`)
  - live matches (`live_1120.jsonl`)
  - pilot 2 (progress report only) and pilot 1 (discarded: gateway cache)
- `results/`: every CSV behind the tables in the report.
- `report/`: LaTeX source, figures and PDF.
- `PREREG.md`, `PREREG_addendum_*.md`: pre-registration and dated deviations.

## Reproduce the tables and figures (no API calls)
    pip install -r requirements.txt     # Python 3.13; exact versions pinned
    export PYTHONPATH=analysis
    python3 analysis/run_main.py      # per-condition fits  -> results/main_fits.csv (Table 2, Appendix C)
    python3 analysis/tests.py         # 67 comparisons      -> results/main_tests.csv
    python3 analysis/run_pooled.py    # pooled fits         -> results/main_pooled.csv
    python3 analysis/run_s3.py        # elicited level 0    -> results/main_s3.csv
    python3 analysis/run_c30.py       # shifted contest     -> results/c30_summary.csv (Table 3)
    python3 analysis/phase2.py results/p2 data/main_haiku.jsonl data/main_gpt.jsonl data/main_gemini.jsonl data/main_llama.jsonl --R 20 --M 300
    python3 analysis/run_live.py      # live matches        -> results/live_summary.csv (Table 4)
    python3 analysis/figs_report.py   # Figures 1-5         -> report/figs/
Run the analysis commands from this folder (the archive root); run the data-collection commands inside harness/. Bootstrap and simulation seeds are fixed (631), so the outputs match the report.

## Collect new data (needs gateway credentials)
    pip install openai
    export TAMU_API_KEY=...   TAMU_COOKIE="CF_Authorization=..."
    cd harness
    python3 harness.py check --models "protected.Claude-Haiku-4.5"
    python3 harness.py run --models <ids> --games beauty req11_20 matrix3x3 --framings neutral ai_competition --cot off on --n 100 --out main.jsonl
    python3 harness.py run --models <ids> --games beauty_p05 beauty_c30 --framings neutral --cot off on --n 100 --out controls.jsonl
    python3 harness.py probe --models <ids> --out nash_probe.jsonl
    python3 elicit_l0.py --models <ids> --n 30 --out elicit_l0.jsonl
    python3 live_1120.py --models <ids> --strategies ne exploit5 exploit0 adaptive --matches 15 --rounds 10 --out live_1120.jsonl
Every command resumes if it is interrupted.

## Reproducibility limits
The models are hosted and may change without notice, so new runs will not reproduce the exact choices. The logs reproduce every reported number exactly. Without the random session code that the harness adds to every call, the gateway returns cached replies.
