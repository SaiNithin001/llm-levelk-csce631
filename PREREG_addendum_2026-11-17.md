## Addendum D4 (2026-11-17), written while the main run is in progress, before any main-run data are seen
This extends `prereg-level0-and-matching.md`. The analysis code was rebuilt on 2026-11-17 as `levelk.py`, `fit.py`, `pooled.py` and `phase2.py`. It reproduces the pilot-2 fits reported in the progress report exactly, for both models.
- **Token caps:** gpt-5-mini gets 8,192 tokens without reasoning and 16,384 with it, because 5 of 20 pilot replies came back empty when hidden reasoning used up the 4,096 cap. Other models are unchanged.
- **Main-run design:**
  - Haiku, GPT-5-mini and Llama: {beauty, 11–20, 3×3} × {neutral, ai_competition} × {reasoning off, on}.
  - Gemini: the same, reasoning off only (D3).
  - p = 1/2 beauty control: neutral framing; reasoning off and on (Gemini off only).
  - 100 calls per cell.
- **Population fit** (progress-report question 1). Each (model, framing, reasoning) cell counts as one player type with equal weight. τ, α and ε are fitted to the pooled distribution per game, with a two-stage bootstrap (resample cells, then 100 choices per cell; B = 1000). Per-cell fits are still reported. Each cell is also labelled by the classic level-k level of its modal choice (`pooled.py`).
- **Phase 2, offline (11–20 game; `phase2.py`).**
  - Opponents: each cell's empirical choice distribution, replayed i.i.d. (this assumes the LLM does not adapt).
  - Matches: 20 rounds, 300 matches per opponent.
  - Exploiter: a Bayesian posterior over the (τ, ε, α) grid of the CH+equilibrium model, uniform prior, updated on the opponent's actions. It plays (1−e)·BR(posterior predictive) + e·NE, with e ∈ {0, 0.1, …, 1}.
  - Payoff: the expected payoff against the opponent's distribution each round. Gain = payoff − payoff of NE play against the same opponent. Interval: bootstrap over matches.
  - Worst case when the estimate is wrong: the same mixture, but best-responding to a fixed CH prediction at a wrong τ′ (τ′ ∈ {0, 0.2, …, 6}, no learning); the minimum gain over τ′ and opponents.
  - Frontier: mean online gain against that worst-case loss as e varies.
  - Baselines: NE, fictitious play, regret matching (full information).
  - Also reported: gain as a function of the number of rounds observed (R = 1, 3, 5, 10, 20).
- **Phase 2, live (`live_1120.py`).** Repeated 11–20 game, 10 rounds, feedback after each round (full history in the prompt), reasoning off. Exploiter strategies: e = 0, e = 0.5, and NE. 5 matches per model for Haiku and GPT-5-mini. Measures the exploiter's realized gain over NE play, and whether the LLM adapts (comparing its requests in rounds 1–3 with rounds 8–10).
