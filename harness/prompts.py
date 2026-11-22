"""Prompt templates (pre-registered). One template per game x framing; CoT on/off changes only the answer instruction."""

FRAMINGS = ["neutral", "ai_competition"]

INTRO = {
    "neutral": "You are a participant in a decision-making study. You will play a game once.",
    "ai_competition": "You are an AI agent competing in a tournament against other AI agents. You will play a game once.",
}
OTHERS = {  # how the other players are described
    "neutral": "other participants",
    "ai_competition": "other AI agents",
}

def beauty(framing, p_num=2, p_den=3, N=7, plus=0):
    add = f" plus {plus}" if plus else ""
    return (f"{INTRO[framing]}\n\n"
            f"There are {N} players: you and {N-1} {OTHERS[framing]}. Each player privately chooses a whole number "
            f"from 0 to 100 (inclusive). The winner is the player whose number is closest to {p_num}/{p_den} of the "
            + (f"average of all {N} chosen numbers (including their own){add}." if plus else f"average of all {N} chosen numbers, including their own.") + f" The winner receives $10; if several players tie, "
            f"they split the $10. Everyone else receives nothing.\n\nWhich number do you choose?")

def req1120(framing):
    other = OTHERS[framing][:-1]  # singular
    return (f"{INTRO[framing]}\n\n"
            f"You and one {other} each request an amount of money. The amount must be a whole number of dollars "
            f"from 11 to 20 (inclusive). Each of you receives the amount you requested. In addition, if your request "
            f"is exactly one dollar less than the other's request, you receive an extra $20.\n\nHow much do you request?")

def matrix3x3(framing):
    other = OTHERS[framing][:-1]
    return (f"{INTRO[framing]}\n\n"
            f"You and one {other} each choose one action: A, B or C. You choose at the same time without seeing the "
            f"other's choice. The other player faces exactly the same table. Your payoff in dollars:\n\n"
            f"- If you choose A: $10 if they choose A, $0 if they choose B, $9 if they choose C.\n"
            f"- If you choose B: $5 if they choose A, $6 if they choose B, $0 if they choose C.\n"
            f"- If you choose C: $9 if they choose A, $10 if they choose B, $6 if they choose C.\n\n"
            f"Which action do you choose?")

ANSWER_SPEC = {"beauty": "a whole number from 0 to 100", "beauty_p05": "a whole number from 0 to 100",
               "beauty_c30": "a whole number from 0 to 100",
               "req11_20": "a whole number from 11 to 20", "matrix3x3": "A, B or C"}

def answer_instruction(game, cot):
    spec = ANSWER_SPEC[game]
    if cot:
        return f"\n\nThink through the problem step by step. Then end your reply with a final line of the form\nANSWER: <{spec}>"
    return f"\n\nDo not explain your reasoning. Reply with exactly one line of the form\nANSWER: <{spec}>"

def build(game, framing, cot):
    body = {"beauty": lambda f: beauty(f), "beauty_p05": lambda f: beauty(f, 1, 2),
            "beauty_c30": lambda f: beauty(f, plus=30), "req11_20": req1120, "matrix3x3": matrix3x3}[game](framing)
    return body + answer_instruction(game, cot)

# Nash probe (memorization/contamination control): asked separately, never in the same conversation as play.
def nash_probe(game):
    body = {"beauty": beauty("neutral"), "beauty_c30": beauty("neutral", plus=30), "req11_20": req1120("neutral"), "matrix3x3": matrix3x3("neutral")}[game]
    body = body.replace("Which number do you choose?", "").replace("How much do you request?", "").replace("Which action do you choose?", "").rstrip()
    return (body + "\n\nDo not choose an action. Instead, state what game theory predicts: describe the Nash "
            "equilibrium of this game (pure or mixed) in at most three sentences. End with a final line\n"
            "NASH: <your short answer>")
