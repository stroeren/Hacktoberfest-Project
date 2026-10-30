"""Champion profiles used to score team comps and to generate training labels."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, fields
from pathlib import Path

ROLES = ("top", "jungle", "mid", "adc", "support")
DMG_TYPES = ("AD", "AP", "mixed")
SCALINGS = ("early", "mid", "late")
RANGES = ("melee", "ranged")
RATED_FIELDS = ("damage", "cc", "frontline", "engage", "peel", "waveclear")


@dataclass(frozen=True)
class Champion:
    name: str
    roles: tuple[str, ...]
    dmg_type: str
    damage: int
    cc: int
    frontline: int
    engage: int
    peel: int
    waveclear: int
    scaling: str
    range: str
    ad_share: float  # physical share of damage dealt (0 = all magic, 1 = all physical)
    stats: dict = field(default_factory=dict, compare=False, hash=False)  # raw numbers, informational

    def damage_split(self) -> tuple[float, float]:
        """Return (AD, AP) damage contribution."""
        return self.damage * self.ad_share, self.damage * (1 - self.ad_share)


def normalize_name(name: str) -> str:
    """'Kai'Sa', 'kaisa' and 'KAI SA' all map to 'kaisa'."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


DMG_TYPE_AD_SHARE = {"AD": 1.0, "AP": 0.0, "mixed": 0.5}
_FIELDS = {f.name for f in fields(Champion)}


class ChampionDB:
    def __init__(self, champions: list[Champion], thresholds: dict | None = None):
        self.champions = champions
        self.thresholds = thresholds or {}  # data-driven issue thresholds; analysis has defaults
        self._by_key = {normalize_name(c.name): c for c in champions}

    @classmethod
    def load(cls, path: str | Path) -> "ChampionDB":
        path = Path(path)
        if not path.exists():
            raise SystemExit(
                f"{path} not found. Build it from Riot data first:\n"
                "  python -m lolcomp.riot.collect\n"
                "  python -m lolcomp.riot.build_champions"
            )
        doc = json.loads(path.read_text(encoding="utf-8"))
        champions = []
        for name, data in doc["champions"].items():
            data = {k: v for k, v in data.items() if k in _FIELDS}
            data.setdefault("ad_share", DMG_TYPE_AD_SHARE.get(data.get("dmg_type"), 0.5))
            champ = Champion(name=name, **{**data, "roles": tuple(data["roles"])})
            _validate(champ)
            champions.append(champ)
        return cls(champions, doc.get("thresholds"))

    def get(self, name: str) -> Champion | None:
        return self._by_key.get(normalize_name(name))

    def for_role(self, role: str) -> list[Champion]:
        return [c for c in self.champions if role in c.roles]


def _validate(c: Champion) -> None:
    problems = []
    if not c.roles or any(r not in ROLES for r in c.roles):
        problems.append(f"roles must be from {ROLES}")
    if c.dmg_type not in DMG_TYPES:
        problems.append(f"dmg_type must be one of {DMG_TYPES}")
    if c.scaling not in SCALINGS:
        problems.append(f"scaling must be one of {SCALINGS}")
    if not 0 <= c.ad_share <= 1:
        problems.append("ad_share must be between 0 and 1")
    if c.range not in RANGES:
        problems.append(f"range must be one of {RANGES}")
    for stat in RATED_FIELDS:
        if getattr(c, stat) not in (0, 1, 2, 3):
            problems.append(f"{stat} must be 0-3")
    if problems:
        raise ValueError(f"Bad champion entry '{c.name}': " + "; ".join(problems))
