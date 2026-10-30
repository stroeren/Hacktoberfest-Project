"""Build champion profiles from fake match summaries shaped like collect.py's output."""

import json
import random

import pytest

from lolcomp.analysis import evaluate_team
from lolcomp.champions import ROLES, ChampionDB
from lolcomp.riot.build_champions import build_profiles, team_thresholds
from lolcomp.riot.collect import summarize

BUILD_CFG = {
    "min_games": 20,
    "role_share": 0.1,
    "short_game_minutes": 25,
    "long_game_minutes": 32,
    "min_games_per_length": 10,
    "issue_quantile": 0.15,
}

# Per-minute stat profiles for fake champions: (physical, magic, taken, cc, ally_heal_shield, cs)
ARCHETYPES = {
    "tank": (150, 150, 2500, 2.0, 0, 6),
    "mage": (20, 900, 600, 0.8, 0, 8),
    "bruiser": (800, 50, 1500, 0.4, 0, 7),
    "marksman": (1000, 30, 500, 0.1, 0, 9),
    "enchanter": (10, 200, 400, 0.6, 400, 1),
}


def fake_champions():
    """Six champions per position, cycling through archetypes. Ids 1-30."""
    champs, ddragon = {}, {}
    cid = 1
    for pos in ROLES:
        for i, arch in enumerate(list(ARCHETYPES) * 2):
            if i == 6:
                break
            champs[cid] = (pos, arch)
            ddragon[cid] = {
                "name": f"{arch.title()}{pos.title()}{i}",
                "tags": [arch],
                "attackrange": 550 if arch in ("mage", "marksman", "enchanter") else 150,
            }
            cid += 1
    return champs, ddragon


def fake_matches(champs, n=600, seed=0):
    rng = random.Random(seed)
    by_pos = {pos: [c for c, (p, _) in champs.items() if p == pos] for pos in ROLES}
    late_scaler = by_pos["mid"][1]  # this champion only wins long games
    matches = []
    for m in range(n):
        duration = rng.randint(15 * 60, 40 * 60)
        minutes = duration / 60
        blue_wins = rng.random() < 0.5
        participants = []
        picks = {pos: rng.sample(by_pos[pos], 2) for pos in ROLES}  # no mirror picks
        for side, team_id in enumerate((100, 200)):
            for pos in ROLES:
                cid = picks[pos][side]
                phys, magic, taken, cc, heal, cs = ARCHETYPES[champs[cid][1]]
                jitter = lambda x: x * minutes * rng.uniform(0.8, 1.2)  # noqa: E731
                participants.append({
                    "champion_id": cid, "team_id": team_id, "position": pos,
                    "win": blue_wins == (team_id == 100),
                    "physical": jitter(phys), "magic": jitter(magic), "true": 0,
                    "taken": jitter(taken), "mitigated": jitter(taken / 2),
                    "cc_time": jitter(cc), "ally_heal_shield": jitter(heal),
                    "minions": jitter(cs), "immobilizations": jitter(cc / 2),
                })
        # Rig the late scaler: it wins exactly the long games.
        for p in participants:
            if p["champion_id"] == late_scaler:
                won = duration >= 32 * 60
                for q in participants:
                    q["win"] = won if q["team_id"] == p["team_id"] else not won
                break
        matches.append({"match_id": f"NA1_{m}", "game_version": "16.21.1.1", "duration": duration,
                        "participants": participants})
    return matches, late_scaler


@pytest.fixture(scope="module")
def built():
    champs, ddragon = fake_champions()
    matches, late_scaler = fake_matches(champs)
    profiles = build_profiles(matches, ddragon, BUILD_CFG)
    return champs, matches, profiles, late_scaler


def by_arch(champs, profiles, arch):
    return [profiles[c] for c, (_, a) in champs.items() if a == arch]


def test_ratings_follow_the_stats(built):
    champs, _, profiles, _ = built
    assert all(p["frontline"] >= 2 for p in by_arch(champs, profiles, "tank"))
    assert all(p["cc"] >= 2 for p in by_arch(champs, profiles, "tank"))
    assert all(p["frontline"] <= 1 for p in by_arch(champs, profiles, "enchanter"))
    assert all(p["peel"] >= 2 for p in by_arch(champs, profiles, "enchanter"))
    assert all(p["dmg_type"] == "AP" and p["ad_share"] < 0.1 for p in by_arch(champs, profiles, "mage"))
    assert all(p["dmg_type"] == "AD" for p in by_arch(champs, profiles, "marksman"))
    assert all(p["range"] == "melee" for p in by_arch(champs, profiles, "tank"))


def test_roles_and_scaling(built):
    champs, _, profiles, late_scaler = built
    for cid, (pos, _) in champs.items():
        assert profiles[cid]["roles"] == [pos]
    assert profiles[late_scaler]["scaling"] == "late"


def test_thresholds_and_db_roundtrip(built, tmp_path):
    _, matches, profiles, _ = built
    thresholds = team_thresholds(matches, profiles, BUILD_CFG["issue_quantile"])
    for stat, (low, high) in thresholds.items():
        assert low < high, stat
    doc = {"thresholds": thresholds,
           "champions": {p["name"]: {k: v for k, v in p.items() if k != "name"} for p in profiles.values()}}
    path = tmp_path / "champions.json"
    path.write_text(json.dumps(doc))
    db = ChampionDB.load(path)
    team = {role: db.for_role(role)[0] for role in ROLES}
    assert evaluate_team(team, db.thresholds).verdict in ("good", "average", "bad")


def test_summarize_skips_remakes_and_other_queues():
    base = {"metadata": {"matchId": "NA1_1"}, "info": {"queueId": 420, "gameDuration": 200,
            "gameVersion": "16.21.1", "participants": []}}
    assert summarize(base) is None
    base["info"].update(gameDuration=1800, queueId=450)
    assert summarize(base) is None
