"""Rule-based team comp analysis.

This is the "teacher" that labels synthetic training data. The fine-tuned Gemma
model learns to reproduce (and later, with hand-labeled data, improve on) these
judgements from champion names alone.

Team totals are sums of five 0-3 ratings, so every total is on a 0-15 scale.
The low/high cutoffs normally come from champions.json, where build_champions
derives them from real teams; the defaults below are only a fallback.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .champions import RATED_FIELDS, ROLES, Champion

DEFAULT_THRESHOLDS = {
    "damage": (7, 13),
    "cc": (5, 11),
    "frontline": (3, 7),
    "engage": (2, 8),
    "peel": (2, 8),
    "waveclear": (5, 11),
    "utility": (9, 36),  # cc + frontline + peel; "too much damage" means high damage with low utility
    "ad_share": (0.2, 0.8),  # team AD share at or below = AP-heavy, at or above = AD-heavy
}

ISSUE_TITLES = {
    "not_enough_damage": "not enough damage",
    "too_much_damage": "too much damage, not enough utility",
    "ad_heavy": "too AD-heavy",
    "ap_heavy": "too AP-heavy",
    "not_enough_cc": "not enough CC",
    "no_frontline": "no frontline",
    "no_engage": "no engage",
    "not_enough_peel": "not enough peel",
    "low_waveclear": "weak waveclear",
    "falls_off": "falls off late",
    "weak_early": "weak early game",
}

VERDICT_CUTOFFS = (("good", 75), ("average", 50), ("bad", 0))


@dataclass
class Issue:
    code: str
    severity: int
    detail: str


@dataclass
class Evaluation:
    verdict: str
    score: int
    ratings: dict[str, str]
    issues: list[Issue] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        """The JSON shape the model is trained to output (severity is internal)."""
        d = asdict(self)
        d["issues"] = [{"code": i.code, "detail": i.detail} for i in self.issues]
        return d


def _rate(total: float, cutoffs: tuple[float, float]) -> str:
    low, high = cutoffs
    if total <= low:
        return "low"
    if total >= high:
        return "high"
    return "ok"


def _join(names: list[str]) -> str:
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def _best_at(champs: list[Champion], stat: str, min_value: int = 2) -> list[str]:
    return [c.name for c in champs if getattr(c, stat) >= min_value]


def evaluate_team(team: dict[str, Champion], thresholds: dict | None = None) -> Evaluation:
    """Evaluate a five-champion team keyed by role.

    thresholds: per-stat (low, high) cutoffs, usually ChampionDB.thresholds.
    """
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    champs = [team[r] for r in ROLES if r in team]
    totals = {stat: sum(getattr(c, stat) for c in champs) for stat in RATED_FIELDS}
    ratings = {stat: _rate(totals[stat], t[stat]) for stat in RATED_FIELDS}

    ad = sum(c.damage_split()[0] for c in champs)
    ap = sum(c.damage_split()[1] for c in champs)
    ad_share = ad / (ad + ap) if ad + ap else 0.5
    if ad_share >= t["ad_share"][1]:
        ratings["damage_mix"] = "AD-heavy"
    elif ad_share <= t["ad_share"][0]:
        ratings["damage_mix"] = "AP-heavy"
    else:
        ratings["damage_mix"] = "balanced"

    early = [c.name for c in champs if c.scaling == "early"]
    late = [c.name for c in champs if c.scaling == "late"]
    if len(early) >= 3 and not late:
        ratings["power_curve"] = "early"
    elif len(late) >= 3:
        ratings["power_curve"] = "late"
    else:
        ratings["power_curve"] = "balanced"

    issues: list[Issue] = []
    carries = _best_at(champs, "damage", 3)
    utility = totals["cc"] + totals["frontline"] + totals["peel"]

    if ratings["damage"] == "low":
        who = f"Only {_join(carries)} can" if carries else "Nobody can"
        issues.append(Issue(
            "not_enough_damage", 35 if totals["damage"] <= 5 else 25,
            f"Team damage is low ({totals['damage']}/15). {who} really threaten kills, "
            "so fights will drag on and the enemy can ignore most of the team.",
        ))
    low_utility = utility <= t["utility"][0]
    if ratings["damage"] == "high" and low_utility:
        issues.append(Issue(
            "too_much_damage", 15,
            f"Almost every pick is a damage dealer ({totals['damage']}/15 damage) with little CC, "
            "frontline or peel to set it up or protect it; the team wins or loses on who gets caught first.",
        ))
    if ratings["damage_mix"] == "AD-heavy":
        issues.append(Issue(
            "ad_heavy", 20,
            f"{round(ad_share * 100)}% of the damage is physical. The enemy can stack armor "
            "and the whole team loses most of its damage.",
        ))
    if ratings["damage_mix"] == "AP-heavy":
        issues.append(Issue(
            "ap_heavy", 20,
            f"{round((1 - ad_share) * 100)}% of the damage is magic. The enemy can stack magic "
            "resist and the whole team loses most of its damage.",
        ))
    if ratings["cc"] == "low":
        sources = _best_at(champs, "cc")
        if not sources:
            who = "nobody brings"
        else:
            who = f"only {_join(sources)} " + ("brings" if len(sources) == 1 else "bring")
        issues.append(Issue(
            "not_enough_cc", 30 if totals["cc"] <= 3 else 20,
            f"Very little crowd control (CC {totals['cc']}/15): {who} reliable lockdown, "
            "so enemy carries and divers can act freely.",
        ))
    if ratings["frontline"] == "low":
        best = max(champs, key=lambda c: c.frontline)
        issues.append(Issue(
            "no_frontline", 20,
            f"No real frontline (frontline {totals['frontline']}/15). Nobody can soak damage for the "
            f"carries; the tankiest pick is {best.name}.",
        ))
    if ratings["engage"] == "low":
        issues.append(Issue(
            "no_engage", 15,
            f"No reliable way to start fights (engage {totals['engage']}/15). The team has to wait "
            "for the enemy to make a mistake or engage first.",
        ))
    if ratings["peel"] == "low":
        issues.append(Issue(
            "not_enough_peel", 10,
            f"Little peel for the backline (peel {totals['peel']}/15). Assassins and divers "
            "will reach the carries easily.",
        ))
    if ratings["waveclear"] == "low":
        issues.append(Issue(
            "low_waveclear", 10,
            f"Weak waveclear ({totals['waveclear']}/15). The team will struggle to defend "
            "sieges and hold side lanes.",
        ))
    if ratings["power_curve"] == "early":
        issues.append(Issue(
            "falls_off", 10,
            f"Built around early-game champions ({_join(early)}) with no late-game scaling; "
            "if the game goes long the team falls off.",
        ))
    if ratings["power_curve"] == "late":
        issues.append(Issue(
            "weak_early", 10,
            f"Several late-game scalers ({_join(late)}); expect to be behind early and to need "
            "to survive until items come online.",
        ))

    strengths: list[str] = []
    if ratings["damage"] == "high" and not low_utility:
        strengths.append(f"High damage ({totals['damage']}/15) led by {_join(carries)}, with enough utility to use it.")
    if ratings["cc"] == "high":
        strengths.append(f"Lots of crowd control ({totals['cc']}/15) from {_join(_best_at(champs, 'cc', 3))}.")
    if ratings["frontline"] == "high":
        strengths.append(f"Strong frontline ({totals['frontline']}/15) from {_join(_best_at(champs, 'frontline', 3))}.")
    if ratings["engage"] == "high":
        strengths.append(f"Plenty of hard engage ({totals['engage']}/15) to force fights.")
    if ratings["peel"] == "high":
        strengths.append(f"Great at protecting carries (peel {totals['peel']}/15).")
    if ratings["waveclear"] == "high":
        strengths.append(f"Strong waveclear ({totals['waveclear']}/15) for sieging and defending.")
    if ratings["damage_mix"] == "balanced" and ratings["damage"] != "low":
        strengths.append(f"Balanced damage profile ({round(ad_share * 100)}% AD / {round((1 - ad_share) * 100)}% AP), hard to itemize against.")

    issues.sort(key=lambda i: i.severity, reverse=True)
    score = max(0, min(100, 100 - sum(i.severity for i in issues)))
    verdict = next(v for v, cutoff in VERDICT_CUTOFFS if score >= cutoff)

    summary = f"{verdict.capitalize()} composition ({score}/100)."
    if issues:
        summary += " Main problems: " + _join([ISSUE_TITLES[i.code] for i in issues[:3]]) + "."
    else:
        summary += " No major weaknesses."
    if strengths and verdict != "bad":
        summary += " " + strengths[0]

    return Evaluation(verdict, score, ratings, issues, strengths, summary)
