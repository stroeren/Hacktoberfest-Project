# LoL Team Comp Evaluator (Gemma 4)

Fine-tunes Gemma 4 with QLoRA to judge a League of Legends team composition and explain what's wrong with it: too much or too little damage, AD/AP-heavy, not enough CC, no frontline, no engage, no peel, weak waveclear, or a lopsided power curve.

```
Top: Fiora / Jungle: Kha'Zix / Mid: Zed / ADC: Draven / Support: Pyke
→ {"verdict": "bad", "score": 0, "issues": [{"code": "not_enough_cc", ...}, {"code": "ad_heavy", ...}, ...]}
```

## How it works

1. **Champion profiles** (`data/champions.json`): a premade, hand-rated file covering ~75 champions. Each one has 0–3 ratings for damage, CC, frontline, engage, peel and waveclear, plus damage type (AD/AP/mixed), scaling and range. Edit it to tune the labels, or add champions in the same format.
2. **Rule-based labeler** (`src/lolcomp/analysis.py`): adds up the team totals (each on a 0–15 scale) and flags issues using the low/high cutoffs in `DEFAULT_THRESHOLDS` at the top of the file. This produces the training labels.
3. **Synthetic dataset** (`generate_data.py`): samples random comps for each role, labels them, and balances good/average/bad.
4. **Fine-tune** (`train.py`): QLoRA SFT with TRL. The model sees only the champion names and learns to output the JSON evaluation, with loss computed on the answer only.
5. **Hand-labeled data** (optional): put your own judgements in `data/labeled/*.jsonl` and list them under `extra_train_files`. They're repeated in training so they count for more than the synthetic rows.

> Trained only on synthetic data, the model learns to copy the rules. Its edge comes from (a) generalizing to champions that aren't in `champions.json`, using what Gemma already knows, and (b) whatever hand-labeled data you add.

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

```powershell
pytest                                          # rules + data tests (no GPU)
python -m lolcomp.generate_data                 # -> data/generated/{train,val}.jsonl
python -m lolcomp.train                         # -> outputs/gemma4-lolcomp (LoRA adapter)
python -m lolcomp.evaluate --limit 200          # verdict accuracy + per-issue precision/recall
python -m lolcomp.evaluate --adapter none       # baseline: untuned Gemma 4
python -m lolcomp.predict --top Ornn --jungle "Lee Sin" --mid Syndra --adc Jinx --support Leona --compare
python -m lolcomp.predict --top Ornn --jungle "Lee Sin" --mid Syndra --adc Jinx --support Leona --rules-only
python -m lolcomp.gui                           # desktop app: pick champions, see rules/model results
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

## Optional: rebuild champions.json from Riot match data

`src/lolcomp/riot/` can replace the premade file with ratings computed from real ranked games. **This overwrites `data/champions.json`.** Run `git checkout data/champions.json` to get the premade version back.

Get a development key at [developer.riotgames.com](https://developer.riotgames.com) (it expires every 24 hours), and set `riot.platform`/`riot.region` in `config.yaml` to your server.

```powershell
$env:RIOT_API_KEY = "RGAPI-..."                 # this terminal session only; never commit it
python -m lolcomp.riot.collect                  # -> data/riot/matches.jsonl (resumable, ~1.5 h for 4000 matches)
python -m lolcomp.riot.build_champions          # -> data/champions.json + win-rate sanity check
```

The built ratings are percentile ranks among champions (bottom quarter = 0, top quarter = 3). The issue cutoffs become the bottom/top 15% of real teams, stored under `thresholds`.

| Field | Source (Riot match-v5 / Data Dragon) |
|---|---|
| `damage` | average share of the team's damage to champions |
| `frontline` | average share of the team's damage taken + self-mitigated |
| `cc` | `timeCCingOthers` per minute |
| `engage` | average of the percentiles for enemy immobilizations/min and frontline |
| `peel` | average of the percentiles for heals + shields on teammates/min and CC |
| `waveclear` | minions/min compared with other champions in the same position |
| `ad_share` / `dmg_type` | physical ÷ (physical + magic) damage to champions |
| `scaling` | win rate in games ≥32 min minus games <25 min; bottom third = early, top third = late |
| `roles` | positions with ≥10% of the champion's games |
| `range` | Data Dragon attack range |
