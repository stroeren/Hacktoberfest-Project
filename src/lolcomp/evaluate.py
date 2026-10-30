"""Score the fine-tuned model on the held-out val set.

    python -m lolcomp.evaluate --config config.yaml --limit 200

Reports JSON validity, verdict accuracy, score error, and precision/recall
of the issue codes (e.g. did it catch "not_enough_cc" when it should have).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from .config import load_config
from .modeling import generate, load_model
from .prompts import parse_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--adapter", help="LoRA adapter dir (default: train.output_dir); 'none' = base model")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    cfg = load_config(args.config)
    with open(f"{cfg['data']['out_dir']}/val.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f][: args.limit]

    adapter = args.adapter or cfg["train"]["output_dir"]
    model, tokenizer = load_model(cfg["model"], adapter=None if adapter == "none" else adapter)

    valid = verdict_hits = 0
    score_err = 0.0
    tp, fp, fn = Counter(), Counter(), Counter()
    for i in range(0, len(rows), args.batch_size):
        batch = rows[i : i + args.batch_size]
        replies = generate(model, tokenizer, [r["prompt"] for r in batch])
        for row, reply in zip(batch, replies):
            gold, pred = row["label"], parse_output(reply)
            if not isinstance(pred, dict):
                fn.update(issue["code"] for issue in gold["issues"])
                continue
            valid += 1
            verdict_hits += pred.get("verdict") == gold["verdict"]
            try:
                score_err += abs(float(pred.get("score", 0)) - gold["score"])
            except (TypeError, ValueError):
                score_err += 100
            gold_codes = {issue["code"] for issue in gold["issues"]}
            pred_codes = {issue.get("code") for issue in pred.get("issues", []) if isinstance(issue, dict)}
            tp.update(gold_codes & pred_codes)
            fp.update(pred_codes - gold_codes)
            fn.update(gold_codes - pred_codes)
        print(f"{min(i + args.batch_size, len(rows))}/{len(rows)}", end="\r")

    n = len(rows)
    print(f"\nvalid JSON:        {valid}/{n} ({valid / n:.1%})")
    print(f"verdict accuracy:  {verdict_hits / n:.1%}")
    print(f"score MAE:         {score_err / max(valid, 1):.1f}")
    print(f"\n{'issue':<20}{'precision':>10}{'recall':>10}{'f1':>8}")
    for code in sorted(set(tp) | set(fp) | set(fn)):
        p = tp[code] / max(tp[code] + fp[code], 1)
        r = tp[code] / max(tp[code] + fn[code], 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        print(f"{code:<20}{p:>10.2f}{r:>10.2f}{f1:>8.2f}")


if __name__ == "__main__":
    main()
