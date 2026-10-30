import json
from pathlib import Path

import pytest

from lolcomp.analysis import evaluate_team
from lolcomp.champions import ROLES, ChampionDB
from lolcomp.prompts import build_example, parse_output

DB = ChampionDB.load(Path(__file__).parent / "fixtures" / "champions.json")


def team(*names):
    return {role: DB.get(name) for role, name in zip(ROLES, names)}


def codes(evaluation):
    return {i.code for i in evaluation.issues}


def test_every_role_has_enough_champions():
    for role in ROLES:
        assert len(DB.for_role(role)) >= 8, role


def test_name_lookup_is_forgiving():
    assert DB.get("kaisa").name == "Kai'Sa"
    assert DB.get("LEE SIN").name == "Lee Sin"
    assert DB.get("not a champ") is None


def test_standard_comp_is_good():
    result = evaluate_team(team("Ornn", "Lee Sin", "Syndra", "Jinx", "Leona"))
    assert result.verdict == "good"
    assert result.issues == []


def test_all_assassins_is_bad():
    result = evaluate_team(team("Fiora", "Kha'Zix", "Zed", "Draven", "Pyke"))
    assert result.verdict == "bad"
    assert {"too_much_damage", "ad_heavy", "not_enough_cc", "no_frontline"} <= codes(result)


def test_all_tanks_lacks_damage():
    result = evaluate_team(team("Ornn", "Sejuani", "Galio", "Ashe", "Alistar"))
    assert "not_enough_damage" in codes(result)
    assert "no_frontline" not in codes(result)


def test_ap_heavy_is_flagged():
    result = evaluate_team(team("Kennen", "Evelynn", "Syndra", "Ziggs", "Lux"))
    assert "ap_heavy" in codes(result)


@pytest.mark.parametrize("reply", [
    '{"verdict": "bad", "score": 10}',
    '```json\n{"verdict": "bad", "score": 10}\n```',
    'Here you go: {"verdict": "bad", "score": 10} hope that helps',
])
def test_parse_output(reply):
    assert parse_output(reply) == {"verdict": "bad", "score": 10}


def test_example_roundtrip():
    names = {r: c.name for r, c in team("Ornn", "Lee Sin", "Syndra", "Jinx", "Leona").items()}
    label = evaluate_team(team(*names.values())).to_dict()
    example = build_example(names, label)
    assert parse_output(example["completion"][0]["content"]) == label
    assert "Lee Sin" in example["prompt"][0]["content"]
    json.dumps(example)  # must be serializable for jsonl
