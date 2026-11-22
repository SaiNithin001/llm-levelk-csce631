## Addendum D5 (2026-11-22): two extensions, decided after the main results, specified before their data
Neither was in the proposal; both are reported as exploratory extensions.

**E1. Beauty contest with an interior equilibrium (`beauty_c30`): a stronger recall-vs-reasoning control.**
- **Game:** N = 7; target = 2/3 × (mean of all 7 guesses) + 30.
- **Equilibrium:** iterated dominance removes guesses below 30 and above 96.7. The continuous game has the unique equilibrium 90; with integer guesses, all choosing 89, 90 or 91 are each symmetric pure equilibria (corrected 2026-11-22 after the data, by checking every unilateral deviation; the "within 3 of 90" criterion below covers all three, so no result changes).
- **Level-k predictions** (uniform level 0, own guess counted in the mean): level 1 = 65, level 2 = 74, level 3 = 80, level 4 = 84; the sequence tends to 90.
- **Design:** neutral framing, reasoning off and on (Gemini off only), 100 calls per cell, plus 3 Nash-statement probes per model.
- **Fit:** the same CH + equilibrium-type model; the equilibrium type is Nash = 90 with the ±3 tolerance.
- **Pre-specified readings:**
  - A cell is called **"recall"** if at least 25% of its guesses are within 1 of 0 (the memorized ⅔-game answer).
  - A cell is called **"equilibrium reasoning"** if at least 25% of its guesses are within 3 of 90.
  - The main comparison is each model's share of equilibrium guesses here against its share of 0 guesses in the standard game.

**E2. Adaptive-ε exploiter (11–20 game, live).**
- **ε each round:** ε_t = P(ADAPTIVE | history) under two hypotheses about the LLM:
  - STATIC: the LLM draws from the CH posterior predictive.
  - ADAPTIVE: the LLM requests the exploiter's previous request minus 1 (20 if that request was 11) with probability 0.7, otherwise uniform.
- **Updating:** prior P(ADAPTIVE) = 0.1. Round 1 has no evidence. Log-odds are clipped to ±20.
- **Play:** (1 − ε_t)·BR(CH posterior) + ε_t·NE, sampled each round.
- **Design:** Haiku, GPT-5-mini and Llama; 15 matches × 10 rounds per model; same prompt and seeding as the earlier live runs.
- **Compared with ε = 0, ε = 0.5 and NE on:**
  - gain over NE play;
  - LLM payoff minus exploiter payoff;
  - the ε trajectory.
- **Hypothesis:** adaptive-ε keeps most of ε = 0's gain against Haiku and Llama, and removes GPT-5-mini's counter-exploitation (LLM − exploiter difference with a 95% CI including 0 or below it).

**Code check:** tested against two scripted fake opponents only; no data was produced.
- An opponent that always undercuts: ε rises 0.1 → 1.0 by round 7, and its payoff falls from 306 to 227 over 10 rounds.
- A constant-16 opponent: ε falls to 0 by round 4.
