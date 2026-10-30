"""Prompt formatting and parsing of the model's JSON answer."""

from __future__ import annotations

import json
import random
import re

from .champions import ROLES

ROLE_LABELS = {"top": "Top", "jungle": "Jungle", "mid": "Mid", "adc": "ADC", "support": "Support"}

INSTRUCTIONS = (
    "You are a League of Legends draft analyst. Evaluate the team composition below. "
    "Judge damage amount, damage mix (AD/AP), crowd control, frontline, engage, peel, "
    "waveclear and power curve. Answer with JSON only, with the keys: "
    "verdict (good/average/bad), score (0-100), ratings, issues (list of {code, detail}), "
    "strengths, summary."
)

# A few phrasings so the model doesn't overfit to one exact prompt.
QUESTION_TEMPLATES = [
    "{team}",
    "Is this a good team comp?\n{team}",
    "Rate my team:\n{team}",
    "Our draft:\n{team}\nWhat's wrong with it?",
]


def format_team(team: dict[str, str]) -> str:
    return "\n".join(f"{ROLE_LABELS[r]}: {team[r]}" for r in ROLES if r in team)


def user_message(team: dict[str, str], rng: random.Random | None = None) -> str:
    template = rng.choice(QUESTION_TEMPLATES) if rng else QUESTION_TEMPLATES[0]
    return INSTRUCTIONS + "\n\n" + template.format(team=format_team(team))


def build_example(team: dict[str, str], label: dict, rng: random.Random | None = None) -> dict:
    """One training row in TRL's conversational prompt/completion format.

    With this format SFTTrainer only computes loss on the completion (the JSON answer).
    """
    return {
        "team": team,
        "label": label,
        "prompt": [{"role": "user", "content": user_message(team, rng)}],
        "completion": [{"role": "assistant", "content": json.dumps(label, ensure_ascii=False)}],
    }


def parse_output(text: str) -> dict | None:
    """Pull the first JSON object out of the model's reply. Returns None if it isn't valid JSON."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
