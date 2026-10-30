"""Generate a labeled train/val set of random team comps.

    python -m lolcomp.generate_data --config config.yaml

Comps are sampled per role from data/champions.json, labeled by the rule-based
analyzer, and balanced so good/average/bad verdicts appear equally often.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from .analysis import evaluate_team
from .champions import RATED_FIELDS, ROLES, Champion, ChampionDB
from .config import load_config
from .prompts import build_example

VERDICTS = ("good", "average", "bad")
# Chance a comp is skewed toward high or low values of one stat. Uniform sampling
# almost never produces e.g. low-damage comps, so the model would rarely see them.
BIAS_PROB = 0.5
BIAS_STATS = (*RATED_FIELDS, "ad", "ap")


def _weight(c: Champion, bias: tuple[str, bool] | None) -> float:
    if bias is None:
        return 1.0
    stat, want_high = bias
    if stat in ("ad", "ap"):
        value = 3 * (c.ad_share if stat == "ad" else 1 - c.ad_share)
    else:
        value = getattr(c, stat)
    return (value + 0.5) ** 2 if want_high else (3.5 - value) ** 2


def random_team(db: ChampionDB, rng: random.Random) -> dict[str, Champion]:
    bias = (rng.choice(BIAS_STATS), rng.random() < 0.5) if rng.random() < BIAS_PROB else None
    team: dict[str, Champion] = {}
    for role in ROLES:
        options = [c for c in db.for_role(role) if c not in team.values()]
        team[role] = rng.choices(options, weights=[_weight(c, bias) for c in options])[0]
    return team


def team_key(team: dict[str, Champion]) -> tuple[str, ...]:
    return tuple(team[r].name for r in ROLES)


def sample_balanced(db: ChampionDB, n: int, rng: random.Random, seen: set) -> list[dict]:
    """Sample n unique comps with roughly equal counts per verdict."""
    quota = {v: n // len(VERDICTS) + (1 if i < n % len(VERDICTS) else 0) for i, v in enumerate(VERDICTS)}
    buckets: dict[str, list[dict]] = {v: [] for v in VERDICTS}
    overflow: list[dict] = []

    for _ in range(n * 500):
        if all(len(buckets[v]) >= quota[v] for v in VERDICTS):
            break
        team = random_team(db, rng)
        key = team_key(team)
        if key in seen:
            continue
        seen.add(key)
        label = evaluate_team(team, db.thresholds).to_dict()
        names = {role: champ.name for role, champ in team.items()}
        example = build_example(names, label, rng)
        bucket = buckets[label["verdict"]]
        (bucket if len(bucket) < quota[label["verdict"]] else overflow).append(example)

    examples = [ex for v in VERDICTS for ex in buckets[v]]
    if len(examples) < n:
        print(f"warning: could not balance verdicts, topping up with {n - len(examples)} extra comps")
        examples += overflow[: n - len(examples)]
    rng.shuffle(examples)
    return examples


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)["data"]
    db = ChampionDB.load(cfg["champions"])
    rng = random.Random(cfg["seed"])
    seen: set = set()

    # Val is sampled from the same pool but never shares a comp with train.
    val = sample_balanced(db, cfg["val_size"], rng, seen)
    train = sample_balanced(db, cfg["train_size"], rng, seen)

    out = Path(cfg["out_dir"])
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "val.jsonl", val)

    for name, rows in (("train", train), ("val", val)):
        verdicts = Counter(r["label"]["verdict"] for r in rows)
        issues = Counter(i["code"] for r in rows for i in r["label"]["issues"])
        print(f"{name}: {len(rows)} comps, verdicts {dict(verdicts)}")
        print(f"  issue counts: {dict(issues.most_common())}")
    print(f"wrote {out / 'train.jsonl'} and {out / 'val.jsonl'}")


if __name__ == "__main__":
    main()
