"""Build data/champions.json from the collected Riot matches.

    python -m lolcomp.riot.build_champions

Every 0-3 rating is a percentile rank among champions with enough games
(bottom quarter = 0 ... top quarter = 3):

  damage     average share of the team's damage to champions
  frontline  average share of the team's damage taken + self-mitigated
  cc         seconds of crowd control applied to enemy champions per minute
  engage     mean of the enemy-immobilizations/min and frontline percentiles
  peel       mean of the heal+shield-on-allies/min and cc percentiles
  waveclear  minions/min compared with other champions in the same position (z-score)

  ad_share   physical / (physical + magic) damage dealt to champions
  scaling    win rate in long games minus win rate in short games:
             bottom third = early, top third = late, rest = mid
  range      Data Dragon attack range

The "too low"/"too high" thresholds for team totals are quantiles of real teams.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev

from ..analysis import evaluate_team
from ..champions import RATED_FIELDS, ROLES, ChampionDB
from ..config import load_config
from .client import ddragon_champions

PER_GAME_METRICS = ("damage", "frontline", "cc", "immob", "ally_heal_shield", "waveclear")
RANGED_MIN_ATTACK_RANGE = 250
AD_TYPE_SHARE = 0.65  # ad_share >= this is labeled "AD", <= 1 - this "AP", otherwise "mixed"


def load_matches(path: str | Path, patches: list[str] | None = None) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"{path} not found. Run `python -m lolcomp.riot.collect` first.")
    with open(path, encoding="utf-8") as f:
        matches = [json.loads(line) for line in f if line.strip()]
    if patches:
        matches = [m for m in matches if patch_of(m) in patches]
    return matches


def patch_of(match: dict) -> str:
    return ".".join(match["game_version"].split(".")[:2])


def _pct_ranks(values: dict) -> dict:
    """Map each key to its percentile rank in [0, 1]; tied values share their average rank."""
    order = sorted(values, key=values.get)
    n = len(order)
    if n == 1:
        return {order[0]: 0.5}
    ranks = {}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in order[i : j + 1]:
            ranks[k] = (i + j) / 2 / (n - 1)
        i = j + 1
    return ranks


def _rating(pct: float) -> int:
    return min(3, int(pct * 4))


def _quantile(values: list[float], q: float) -> float:
    xs = sorted(values)
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def aggregate(matches: list[dict], short_s: int, long_s: int):
    """Sum per-game stats for every champion."""
    cs_by_pos = defaultdict(list)
    for m in matches:
        for p in m["participants"]:
            cs_by_pos[p["position"]].append(p["minions"] / (m["duration"] / 60))
    cs_norm = {pos: (mean(v), pstdev(v) or 1.0) for pos, v in cs_by_pos.items()}

    acc: dict[int, Counter] = defaultdict(Counter)
    roles: dict[int, Counter] = defaultdict(Counter)
    for m in matches:
        minutes = m["duration"] / 60
        team_dmg, team_tank = Counter(), Counter()
        for p in m["participants"]:
            team_dmg[p["team_id"]] += p["physical"] + p["magic"] + p["true"]
            team_tank[p["team_id"]] += p["taken"] + p["mitigated"]
        for p in m["participants"]:
            a = acc[p["champion_id"]]
            roles[p["champion_id"]][p["position"]] += 1
            a["games"] += 1
            a["wins"] += p["win"]
            a["damage"] += (p["physical"] + p["magic"] + p["true"]) / max(team_dmg[p["team_id"]], 1)
            a["frontline"] += (p["taken"] + p["mitigated"]) / max(team_tank[p["team_id"]], 1)
            a["cc"] += p["cc_time"] / minutes
            a["ally_heal_shield"] += p["ally_heal_shield"] / minutes
            cs_mean, cs_std = cs_norm[p["position"]]
            a["waveclear"] += (p["minions"] / minutes - cs_mean) / cs_std
            if p.get("immobilizations") is not None:
                a["immob"] += p["immobilizations"] / minutes
                a["immob_games"] += 1
            a["physical"] += p["physical"]
            a["magic"] += p["magic"]
            if m["duration"] < short_s:
                a["short_games"] += 1
                a["short_wins"] += p["win"]
            elif m["duration"] >= long_s:
                a["long_games"] += 1
                a["long_wins"] += p["win"]
    return acc, roles


def build_profiles(matches: list[dict], ddragon: dict[int, dict], cfg: dict) -> dict[int, dict]:
    """Champion profiles keyed by champion id."""
    acc, roles = aggregate(matches, cfg["short_game_minutes"] * 60, cfg["long_game_minutes"] * 60)
    eligible = [cid for cid, a in acc.items() if a["games"] >= cfg["min_games"] and cid in ddragon]
    if len(eligible) < 10:
        raise SystemExit(f"Only {len(eligible)} champions have {cfg['min_games']}+ games; collect more matches.")

    avg = {}
    for cid in eligible:
        a = acc[cid]
        avg[cid] = {k: a[k] / a["games"] for k in PER_GAME_METRICS if k != "immob"}
        avg[cid]["immob"] = a["immob"] / a["immob_games"] if a["immob_games"] else 0.0

    pct = {k: _pct_ranks({cid: avg[cid][k] for cid in eligible}) for k in PER_GAME_METRICS}
    pct["engage"] = _pct_ranks({cid: (pct["immob"][cid] + pct["frontline"][cid]) / 2 for cid in eligible})
    pct["peel"] = _pct_ranks({cid: (pct["ally_heal_shield"][cid] + pct["cc"][cid]) / 2 for cid in eligible})

    min_bucket = cfg["min_games_per_length"]
    deltas = {
        cid: acc[cid]["long_wins"] / acc[cid]["long_games"] - acc[cid]["short_wins"] / acc[cid]["short_games"]
        for cid in eligible
        if acc[cid]["short_games"] >= min_bucket and acc[cid]["long_games"] >= min_bucket
    }
    delta_pct = _pct_ranks(deltas)

    profiles = {}
    for cid in eligible:
        a, dd = acc[cid], ddragon[cid]
        dmg_ad_ap = a["physical"] + a["magic"]
        ad_share = a["physical"] / dmg_ad_ap if dmg_ad_ap else 0.5
        role_counts = roles[cid]
        champ_roles = [r for r in ROLES if role_counts[r] / a["games"] >= cfg["role_share"]]
        if cid not in delta_pct:
            scaling = "mid"
        else:
            scaling = "early" if delta_pct[cid] < 1 / 3 else "late" if delta_pct[cid] >= 2 / 3 else "mid"
        profiles[cid] = {
            "name": dd["name"],
            "roles": champ_roles or [role_counts.most_common(1)[0][0]],
            "dmg_type": "AD" if ad_share >= AD_TYPE_SHARE else "AP" if ad_share <= 1 - AD_TYPE_SHARE else "mixed",
            "ad_share": round(ad_share, 3),
            **{k: _rating(pct[k][cid]) for k in RATED_FIELDS},
            "scaling": scaling,
            "range": "ranged" if dd["attackrange"] > RANGED_MIN_ATTACK_RANGE else "melee",
            "stats": {
                "games": a["games"],
                "win_rate": round(a["wins"] / a["games"], 4),
                "role_games": {r: role_counts[r] for r in ROLES if role_counts[r]},
                "long_minus_short_wr": round(deltas[cid], 4) if cid in deltas else None,
                "tags": dd["tags"],
                **{f"avg_{k}": round(v, 4) for k, v in avg[cid].items()},
            },
        }
    return profiles


def team_thresholds(matches: list[dict], profiles: dict[int, dict], q: float) -> dict:
    """Low/high cutoffs for team totals, taken from the real teams in the data."""
    totals = defaultdict(list)
    ad_shares = []
    for m in matches:
        for team_id in {p["team_id"] for p in m["participants"]}:
            team = [profiles.get(p["champion_id"]) for p in m["participants"] if p["team_id"] == team_id]
            if len(team) != 5 or None in team:
                continue
            for stat in RATED_FIELDS:
                totals[stat].append(sum(c[stat] for c in team))
            totals["utility"].append(sum(c["cc"] + c["frontline"] + c["peel"] for c in team))
            dmg = sum(c["damage"] for c in team)
            ad_shares.append(sum(c["damage"] * c["ad_share"] for c in team) / dmg if dmg else 0.5)
    if not ad_shares:
        raise SystemExit("No complete teams found to compute thresholds; collect more matches.")

    thresholds = {}
    for stat, values in totals.items():
        low, high = round(_quantile(values, q)), round(_quantile(values, 1 - q))
        thresholds[stat] = [low, max(high, low + 1)]
    thresholds["ad_share"] = [round(_quantile(ad_shares, q), 3), round(_quantile(ad_shares, 1 - q), 3)]
    return thresholds


def report_win_rates(matches: list[dict], db: ChampionDB, id_to_name: dict[int, str]) -> None:
    """Sanity check: do the rules' verdicts and issues line up with real wins and losses?"""
    by_verdict, by_issue = defaultdict(Counter), defaultdict(Counter)
    for m in matches:
        for team_id in {p["team_id"] for p in m["participants"]}:
            players = [p for p in m["participants"] if p["team_id"] == team_id]
            team = {p["position"]: db.get(id_to_name.get(p["champion_id"], "")) for p in players}
            if len(team) != 5 or None in team.values():
                continue
            result = evaluate_team(team, db.thresholds)
            won = players[0]["win"]
            by_verdict[result.verdict].update(games=1, wins=won)
            for issue in result.issues:
                by_issue[issue.code].update(games=1, wins=won)

    print("\nreal win rate by rule verdict (good should be highest):")
    for verdict in ("good", "average", "bad"):
        c = by_verdict[verdict]
        if c["games"]:
            print(f"  {verdict:<8} {c['wins'] / c['games']:.1%}  ({c['games']} teams)")
    print("real win rate of teams with each issue:")
    for code, c in sorted(by_issue.items(), key=lambda kv: kv[1]["wins"] / kv[1]["games"]):
        print(f"  {code:<18} {c['wins'] / c['games']:.1%}  ({c['games']} teams)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    bcfg = cfg["build"]
    matches = load_matches(cfg["riot"]["matches_file"], bcfg.get("patches"))
    print(f"building from {len(matches)} matches (patches: {sorted({patch_of(m) for m in matches})})")

    ddragon_version, ddragon = ddragon_champions()
    profiles = build_profiles(matches, ddragon, bcfg)
    thresholds = team_thresholds(matches, profiles, bcfg["issue_quantile"])

    doc = {
        "_source": (
            f"Built by lolcomp.riot.build_champions from {len(matches)} ranked solo/duo matches "
            f"(Riot match-v5) and Data Dragon {ddragon_version}. Ratings are percentile ranks; "
            "do not edit by hand, rebuild instead."
        ),
        "patches": sorted({patch_of(m) for m in matches}),
        "thresholds": thresholds,
        "champions": {
            p["name"]: {k: v for k, v in p.items() if k != "name"}
            for p in sorted(profiles.values(), key=lambda p: p["name"])
        },
    }
    out = Path(cfg["data"]["champions"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")

    roles = Counter(r for p in profiles.values() for r in p["roles"])
    print(f"wrote {len(profiles)} champions to {out}  (per role: {dict(roles)})")
    print(f"thresholds: {thresholds}")
    report_win_rates(matches, ChampionDB.load(out), {cid: p["name"] for cid, p in profiles.items()})


if __name__ == "__main__":
    main()
