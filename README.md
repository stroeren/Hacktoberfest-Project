# LoL Team Comp Evaluator (Gemma 4)

Fine-tunes Gemma 4 with QLoRA to judge a League of Legends team composition and explain what's wrong with it: too much or too little damage, AD/AP-heavy, not enough CC, no frontline, no engage, no peel, weak waveclear, or a lopsided power curve.

```
Top: Fiora / Jungle: Kha'Zix / Mid: Zed / ADC: Draven / Support: Pyke
→ {"verdict": "bad", "score": 0, "issues": [{"code": "not_enough_cc", ...}, {"code": "ad_heavy", ...}, ...]}
```

## How it works

1. **Champion profiles from Riot data** (`src/lolcomp/riot/`): `collect` downloads ranked solo/duo matches from the Riot API (match-v5). `build_champions` turns them into `data/champions.json`. Every rating is a percentile rank among champions, not a hand judgement (see table below).
2. **Rule-based labeler** (`src/lolcomp/analysis.py`): adds up the team totals and flags issues. The "too low"/"too high" cutoffs are the bottom/top 15% of real teams in your match data, stored in `champions.json` under `thresholds`. This produces the training labels.
3. **Synthetic dataset** (`generate_data.py`): samples random comps for each role, labels them, and balances good/average/bad.
4. **Fine-tune** (`train.py`): QLoRA SFT with TRL. The model sees only the champion names and learns to output the JSON evaluation, with loss computed on the answer only.
5. **Hand-labeled data** (optional): put your own judgements in `data/labeled/*.jsonl` and list them under `extra_train_files`. They're repeated in training so they count for more than the synthetic rows.

### Where each champion stat comes from

Ratings are 0–3 by quartile among all champions with at least `min_games` games (bottom 25% = 0, top 25% = 3).

| Field | Source (Riot match-v5 / Data Dragon) |
|---|---|
| `damage` | average share of the team's damage to champions |
| `frontline` | average share of the team's damage taken + self-mitigated |
| `cc` | `timeCCingOthers` per minute |
| `engage` | average of the percentiles for enemy immobilizations/min (`challenges.enemyChampionImmobilizations`) and frontline |
| `peel` | average of the percentiles for heals + shields on teammates/min and CC |
| `waveclear` | minions/min compared with other champions in the same position (z-score) |
| `ad_share` / `dmg_type` | physical ÷ (physical + magic) damage to champions |
| `scaling` | win rate in games ≥32 min minus games <25 min; bottom third = early, top third = late |
| `roles` | positions with ≥10% of the champion's games |
| `range` | Data Dragon attack range |

Riot doesn't track engage or peel directly, so those two are proxies built from the stats above. The raw numbers behind every rating are kept under each champion's `stats` key. The build also prints the real win rate of good/average/bad teams and of teams with each issue, so you can check whether the rules actually line up with winning.

> Trained only on synthetic data, the model learns to copy the rules. Its edge comes from (a) generalizing to champions that aren't in `champions.json`, using what Gemma already knows, and (b) whatever hand-labeled or match-outcome data you add.

## Setup (Windows, NVIDIA GPU)

```powershell
python -m venv .venv
.venv\Scripts\activate
# PyTorch with CUDA. Pick the command for your CUDA version at https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[train,dev]"
hf auth login        # Gemma is gated: accept the license on its Hugging Face page first
```

VRAM: `gemma-4-E4B-it` in 4-bit needs roughly 12–16 GB for training. On a smaller GPU, switch `base_model` to a smaller Gemma 4 variant or lower `max_length`/`batch_size`.

## Usage

### 1. Build champion data from the Riot API

Get a development key at [developer.riotgames.com](https://developer.riotgames.com). It expires every 24 hours. Set `platform`/`region` in `config.yaml` to your server.

```powershell
$env:RIOT_API_KEY = "RGAPI-..."                 # this terminal session only; never commit it
python -m lolcomp.riot.collect                  # -> data/riot/matches.jsonl (resumable, ~1.5 h for 4000 matches)
python -m lolcomp.riot.build_champions          # -> data/champions.json + win-rate sanity check
```

The collector is rate-limited to development-key limits (100 requests per 2 minutes). Stop it any time and rerun to continue. Set `build.patches` to restrict the build to recent patches once you have enough matches.

### 2. Train and use the model

```powershell
pytest                                          # rules + build logic tests (no GPU, no API key)
python -m lolcomp.generate_data                 # -> data/generated/{train,val}.jsonl
python -m lolcomp.train                         # -> outputs/gemma4-lolcomp (LoRA adapter)
python -m lolcomp.evaluate --limit 200          # verdict accuracy + per-issue precision/recall
python -m lolcomp.evaluate --adapter none       # baseline: untuned Gemma 4
python -m lolcomp.predict --top Ornn --jungle "Lee Sin" --mid Syndra --adc Jinx --support Leona --compare
python -m lolcomp.predict --top Ornn --jungle "Lee Sin" --mid Syndra --adc Jinx --support Leona --rules-only
```

All settings are in `config.yaml`.

## Output format

```json
{
  "verdict": "good | average | bad",
  "score": 0-100,
  "ratings": {"damage": "low|ok|high", "cc": "...", "frontline": "...", "engage": "...",
              "peel": "...", "waveclear": "...", "damage_mix": "AD-heavy|AP-heavy|balanced",
              "power_curve": "early|balanced|late"},
  "issues": [{"code": "not_enough_cc", "detail": "..."}],
  "strengths": ["..."],
  "summary": "..."
}
```

Issue codes: `not_enough_damage`, `too_much_damage`, `ad_heavy`, `ap_heavy`, `not_enough_cc`, `no_frontline`, `no_engage`, `not_enough_peel`, `low_waveclear`, `falls_off`, `weak_early`.
