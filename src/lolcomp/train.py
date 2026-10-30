"""QLoRA fine-tune Gemma 4 on the generated comp evaluations.

    python -m lolcomp.train --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import random

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from .config import load_config
from .modeling import compute_dtype, load_model
from .prompts import build_example


def load_extra_examples(paths: list[str], seed: int) -> list[dict]:
    """Hand-labeled comps: one {"team": {...}, "label": {...}} object per line."""
    rng = random.Random(seed)
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    rows.append(build_example(row["team"], row["label"], rng))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg, lora_cfg, train_cfg = cfg["data"], cfg["lora"], cfg["train"]

    columns = ["prompt", "completion"]
    files = {"train": f"{data_cfg['out_dir']}/train.jsonl", "val": f"{data_cfg['out_dir']}/val.jsonl"}
    ds = load_dataset("json", data_files=files)
    train_ds, val_ds = ds["train"].select_columns(columns), ds["val"].select_columns(columns)

    extra = load_extra_examples(data_cfg.get("extra_train_files") or [], data_cfg["seed"])
    if extra:
        # Hand labels are worth more than synthetic ones, so repeat them.
        repeat = data_cfg.get("extra_train_repeat", 3)
        extra_ds = Dataset.from_list([{k: r[k] for k in columns} for r in extra] * repeat)
        train_ds = concatenate_datasets([train_ds, extra_ds]).shuffle(seed=data_cfg["seed"])
        print(f"added {len(extra)} hand-labeled comps (x{repeat})")

    model, tokenizer = load_model(cfg["model"])

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type="CAUSAL_LM",
    )

    bf16 = compute_dtype() == torch.bfloat16
    sft_config = SFTConfig(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["epochs"],
        learning_rate=train_cfg["learning_rate"],
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["grad_accum"],
        max_length=train_cfg["max_length"],
        lr_scheduler_type="cosine",
        warmup_steps=train_cfg["warmup_steps"],
        gradient_checkpointing=True,
        bf16=bf16,
        fp16=not bf16,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=train_cfg["eval_steps"],
        save_strategy="steps",
        save_steps=train_cfg["eval_steps"],
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        seed=data_cfg["seed"],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.model.print_trainable_parameters()
    trainer.train()

    trainer.save_model(train_cfg["output_dir"])  # saves the LoRA adapter only
    tokenizer.save_pretrained(train_cfg["output_dir"])
    print(f"adapter saved to {train_cfg['output_dir']}")


if __name__ == "__main__":
    main()
