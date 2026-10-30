"""Collect ranked solo/duo matches from the Riot API.

    $env:RIOT_API_KEY = "RGAPI-..."        # PowerShell
    python -m lolcomp.riot.collect

Seeds players from the ranked ladder tiers in config.yaml, pulls their recent
matches, and appends a compact summary of each match to riot.matches_file.
It's safe to stop and rerun: matches already in the file are skipped.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ..config import load_config
from .client import RANKED_SOLO_QUEUE_ID, RiotClient

POSITIONS = {"TOP": "top", "JUNGLE": "jungle", "MIDDLE": "mid", "BOTTOM": "adc", "UTILITY": "support"}
MIN_DURATION_S = 600  # drop remakes and very early surrenders


def summarize(match: dict) -> dict | None:
    """Keep only the fields used to build champion profiles (plus the result, for later)."""
    info = match["info"]
    if info.get("queueId") != RANKED_SOLO_QUEUE_ID or info.get("gameDuration", 0) < MIN_DURATION_S:
        return None
    participants = []
    for p in info["participants"]:
        position = POSITIONS.get(p.get("teamPosition"))
        if position is None:
            return None
        challenges = p.get("challenges") or {}
        participants.append({
            "champion_id": p["championId"],
            "team_id": p["teamId"],
            "position": position,
            "win": p["win"],
            "physical": p["physicalDamageDealtToChampions"],
            "magic": p["magicDamageDealtToChampions"],
            "true": p["trueDamageDealtToChampions"],
            "taken": p["totalDamageTaken"],
            "mitigated": p["damageSelfMitigated"],
            "cc_time": p["timeCCingOthers"],
            "ally_heal_shield": p["totalHealsOnTeammates"] + p["totalDamageShieldedOnTeammates"],
            "minions": p["totalMinionsKilled"],
            "immobilizations": challenges.get("enemyChampionImmobilizations"),
        })
    return {
        "match_id": match["metadata"]["matchId"],
        "game_version": info["gameVersion"],
        "duration": info["gameDuration"],
        "participants": participants,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)["riot"]
    client = RiotClient(cfg["platform"], cfg["region"])
    out = Path(cfg["matches_file"])
    out.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    if out.exists():
        with open(out, encoding="utf-8") as f:
            seen = {json.loads(line)["match_id"] for line in f if line.strip()}
    saved = len(seen)
    print(f"{saved} matches already collected, target {cfg['max_matches']}")

    rng = random.Random(cfg.get("seed", 0))
    puuids: list[str] = []
    for tier in cfg["tiers"]:
        tier_name, division = (tier.split() + [None])[:2]
        found = client.league_puuids(cfg["queue"], tier_name, division)
        rng.shuffle(found)
        puuids += found[: cfg["players_per_tier"]]
        print(f"seeded {min(len(found), cfg['players_per_tier'])} players from {tier}")
    rng.shuffle(puuids)

    with open(out, "a", encoding="utf-8") as f:
        for i, puuid in enumerate(puuids, 1):
            if saved >= cfg["max_matches"]:
                break
            for match_id in client.match_ids(puuid, cfg["matches_per_player"]):
                if match_id in seen or saved >= cfg["max_matches"]:
                    continue
                seen.add(match_id)
                match = client.match(match_id)
                summary = summarize(match) if match else None
                if summary:
                    f.write(json.dumps(summary) + "\n")
                    f.flush()
                    saved += 1
            print(f"player {i}/{len(puuids)}: {saved}/{cfg['max_matches']} matches", flush=True)

    print(f"done: {saved} matches in {out}")


if __name__ == "__main__":
    main()
