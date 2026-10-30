"""Minimal Riot API client: rate limiting, retries, and the few endpoints we need.

The API key is read from the RIOT_API_KEY environment variable so it never ends
up in config files or git.
"""

from __future__ import annotations

import os
import time
from collections import deque

import requests

# Development keys allow 20 requests/second and 100 requests/2 minutes.
DEV_KEY_LIMITS = ((20, 1.0), (100, 120.0))
RANKED_SOLO_QUEUE_ID = 420
APEX_TIERS = {"CHALLENGER": "challengerleagues", "GRANDMASTER": "grandmasterleagues", "MASTER": "masterleagues"}


class RateLimiter:
    def __init__(self, limits=DEV_KEY_LIMITS):
        self.limits = limits
        self.calls: deque[float] = deque()

    def wait(self) -> None:
        longest = max(window for _, window in self.limits)
        while True:
            now = time.monotonic()
            while self.calls and now - self.calls[0] > longest:
                self.calls.popleft()
            sleep_for = 0.0
            for limit, window in self.limits:
                recent = [t for t in self.calls if now - t < window]
                if len(recent) >= limit:
                    # Wait until the call that pushed us over the limit leaves the window.
                    sleep_for = max(sleep_for, window - (now - recent[-limit]) + 0.05)
            if sleep_for <= 0:
                self.calls.append(now)
                return
            time.sleep(sleep_for)


class RiotClient:
    def __init__(self, platform: str, region: str, api_key: str | None = None, limits=DEV_KEY_LIMITS):
        key = api_key or os.environ.get("RIOT_API_KEY")
        if not key:
            raise SystemExit(
                "Set the RIOT_API_KEY environment variable first "
                "(get a key at https://developer.riotgames.com)."
            )
        self.platform_url = f"https://{platform}.api.riotgames.com"
        self.region_url = f"https://{region}.api.riotgames.com"
        self.session = requests.Session()
        self.session.headers["X-Riot-Token"] = key
        self.limiter = RateLimiter(limits)

    def _get(self, url: str, params: dict | None = None):
        for attempt in range(6):
            self.limiter.wait()
            try:
                r = self.session.get(url, params=params, timeout=30)
            except requests.RequestException as e:
                print(f"  network error ({e}), retrying")
                time.sleep(2**attempt)
                continue
            if r.status_code == 429:
                time.sleep(int(r.headers.get("Retry-After", 10)))
                continue
            if r.status_code >= 500:
                time.sleep(2**attempt)
                continue
            if r.status_code == 404:
                return None
            if r.status_code in (401, 403):
                raise SystemExit(
                    f"Riot API rejected the key (HTTP {r.status_code}). Development keys expire "
                    "every 24 hours; generate a new one at https://developer.riotgames.com."
                )
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"giving up on {url}")

    def league_puuids(self, queue: str, tier: str, division: str | None = None, page: int = 1) -> list[str]:
        """PUUIDs of ranked players in a tier (apex tiers ignore division/page)."""
        tier = tier.upper()
        if tier in APEX_TIERS:
            data = self._get(f"{self.platform_url}/lol/league/v4/{APEX_TIERS[tier]}/by-queue/{queue}")
            entries = (data or {}).get("entries", [])
        else:
            entries = self._get(
                f"{self.platform_url}/lol/league/v4/entries/{queue}/{tier}/{division}", {"page": page}
            ) or []
        return [e["puuid"] for e in entries if e.get("puuid")]

    def match_ids(self, puuid: str, count: int = 20, queue_id: int = RANKED_SOLO_QUEUE_ID) -> list[str]:
        return self._get(
            f"{self.region_url}/lol/match/v5/matches/by-puuid/{puuid}/ids",
            {"queue": queue_id, "type": "ranked", "start": 0, "count": count},
        ) or []

    def match(self, match_id: str) -> dict | None:
        return self._get(f"{self.region_url}/lol/match/v5/matches/{match_id}")


def ddragon_champions() -> tuple[str, dict[int, dict]]:
    """Latest Data Dragon champion data keyed by numeric champion id. No API key needed."""
    version = requests.get("https://ddragon.leagueoflegends.com/api/versions.json", timeout=30).json()[0]
    data = requests.get(
        f"https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion.json", timeout=30
    ).json()["data"]
    return version, {
        int(c["key"]): {"name": c["name"], "tags": c["tags"], "attackrange": c["stats"]["attackrange"]}
        for c in data.values()
    }
