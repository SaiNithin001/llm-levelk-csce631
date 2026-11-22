# Pre-registration: level-0, level-matching and fitting rules
CSCE 631-D, Fall 2026 · Sai Maddi (UIN 435003987) · written 2026-11-04, before any Phase 1 LLM data was collected.
The original validation scripts (synthetic-agent recovery, coverage and level-0 misspecification) were lost when the working environment was reset. The model and fitting code were rebuilt on 2026-11-17 as `analysis/levelk.py` and `analysis/fit.py`; the rebuilt code reproduces the pilot-2 fits exactly (see `PREREG_addendum_2026-11-17.md`). After data collection started, changes were made only in the dated deviations log below and in the addenda.

## 1. Games
- **Beauty contest.** N = 7, integer guesses 0–100, target = p × (mean of all N guesses, own included), p = 2/3. Neutral-framing replication with p = 1/2 as the memorization control. Nash = 0.
- **11–20 game.** Choose an integer r in 11..20; the payoff is r, plus 20 if r is exactly one below the other player's request.
  - Level 1 = 19.
  - The unique symmetric Nash equilibrium is mixed: requests 15–20 with probabilities .25, .25, .20, .15, .10, .05, value 20.
- **3×3 game.** Symmetric, row payoffs [[10,0,9],[5,6,0],[9,10,6]] for A, B, C. Iterated strict dominance removes B, then C, leaving A. Level 1 = C; every level ≥ 2 = A.

## 2. Level 0
Level 0 plays uniformly at random over the action set and is a genuine type with Poisson weight e^{−τ}.
Sensitivity analyses:
- **S2:** level 0 = request 20 in the 11–20 game.
- **S3:** level 0 elicited from each model.

A τ finding is called robust only if it holds under the primary definition and S2/S3; otherwise it is reported as level-0-dependent.

## 3. Levels k ≥ 1
- **Weights:** Poisson weights over k = 0..10, renormalized.
- **Beliefs:** level k best-responds to levels 0..k−1 in proportions f(h)/Σ_{l<k} f(l).
- **Ties:** exact best-response ties are split uniformly.
- **Tolerance:** beauty-contest predictions carry a discrete normal tolerance with σ = 3; the other games are exact.

## 4. Choices matching several levels or none
P(a | τ, ε) = (1−ε) Σ_k f_τ(k) L_k(a) + ε·Uniform(a), with ε ≤ 0.5 estimated per cell.
- **Several levels:** their weighted contributions add up.
- **No level:** level 0 and the tremble explain the choice; nothing is dropped.
- **Unparseable replies:** excluded and counted.
- **S1:** ε = 0 as a sensitivity analysis.

## 5. Estimation
- **Grid:** maximum likelihood on τ ∈ {0, 0.05, …, 6} and ε ∈ {0, 0.05, …, 0.5}.
- **Intervals:** percentile bootstrap, B = 2000, 95%.
- **Grid edge:** an estimate or upper bound at 6.0 is reported as not identified.
- **Hypotheses:**
  - H1: model pairs differ.
  - H2: reasoning and framing effects within a model and game.
  - H3: an ε-safe gain over equilibrium play at bounded worst-case loss.
- **Multiple comparisons:** no correction; the tests are descriptive.

## 6. Validation before data (synthetic agents, 100 choices per cell)
- **Recovery:** with ε free, bias was within ±0.1 in the beauty and 11–20 games.
- **Coverage:** 95% intervals covered the true τ in 95–97% of simulations.
- **Fixed ε:** with ε fixed at 0, τ was underestimated.
- **Level-0 misspecification in the 11–20 game:** for low τ, a true focal level 0 of 20 pushes the uniform-level-0 fit to the grid edge.

## Deviations log
- **D1, after pilot 1 (Nov 4):**
  - Random session code on every call, because the gateway cached replies.
  - The parser accepts a "$" before the number.
  - Higher token limits (4096 without reasoning, 8192 with it).
  - Pilot 1 discarded.
- **D2, after pilot 2 (Nov 4):** equilibrium type α added, P(a) = (1−ε−α)·CH_τ(a) + α·Nash(a) + ε/|A|. The pre-registered model is reported alongside.
- **D3, after pilot 2:** Gemini reasoning-on cells dropped; GPT-5-mini and Gemini treated as reasoning internally.
- **D4 (Nov 17):** main-run design, GPT-5-mini token limits, population fit, Phase 2 protocols; see `PREREG_addendum_2026-11-17.md`.
- **D5 (Nov 22):** the shifted beauty contest and the adaptive-ε exploiter; see `PREREG_addendum_D5_2026-11-22.md`.
