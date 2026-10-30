"""Evaluate one team comp with the fine-tuned model (or just the rules).

    python -m lolcomp.predict --top Ornn --jungle "Lee Sin" --mid Syndra --adc Jinx --support Leona
    python -m lolcomp.predict ... --rules-only     # no GPU needed, rule-based answer
    python -m lolcomp.predict ... --compare        # model answer + rule-based answer
"""

from __future__ import annotations

import argparse
import json

from .analysis import evaluate_team
from .champions import ROLES, ChampionDB
from .config import load_config
from .prompts import parse_output, user_message


def rules_answer(db: ChampionDB, team: dict[str, str]) -> dict | None:
    champs = {role: db.get(name) for role, name in team.items()}
    unknown = [team[r] for r, c in champs.items() if c is None]
    if unknown:
        print(f"(rules skipped: not in champions.json: {', '.join(unknown)})")
        return None
    return evaluate_team(champs, db.thresholds).to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    for role in ROLES:
        parser.add_argument(f"--{role}", required=True)
    parser.add_argument("--adapter", help="LoRA adapter dir (default: train.output_dir from config)")
    parser.add_argument("--rules-only", action="store_true")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    db = ChampionDB.load(cfg["data"]["champions"])
    # Use canonical spelling for known champions; pass unknown ones through for the model.
    team = {}
    for role in ROLES:
        name = getattr(args, role)
        champ = db.get(name)
        team[role] = champ.name if champ else name

    if args.rules_only or args.compare:
        print("=== rule-based ===")
        print(json.dumps(rules_answer(db, team), indent=2))
        if args.rules_only:
            return

    from .modeling import generate, load_model  # heavy imports only when needed

    model, tokenizer = load_model(cfg["model"], adapter=args.adapter or cfg["train"]["output_dir"])
    [reply] = generate(model, tokenizer, [[{"role": "user", "content": user_message(team)}]])
    parsed = parse_output(reply)
    print("=== model ===")
    print(json.dumps(parsed, indent=2) if parsed else f"(could not parse JSON)\n{reply}")


if __name__ == "__main__":
    main()
