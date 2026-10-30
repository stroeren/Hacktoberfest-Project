"""Loading Gemma 4 (optionally 4-bit + LoRA adapter) and batched generation."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def compute_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32  # most CPUs have no fast bf16/fp16 matmuls
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_model(model_cfg: dict, adapter: str | None = None):
    """Load the base model from config, plus a trained LoRA adapter if given."""
    on_gpu = torch.cuda.is_available()
    if not on_gpu:
        print("No CUDA GPU found: running on CPU in float32. This is slow; use a small model (see config.cpu.yaml).")

    quant = None
    if on_gpu and model_cfg.get("load_in_4bit", True):
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype(),
        )

    tokenizer = AutoTokenizer.from_pretrained(adapter or model_cfg["base_model"])
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model"],
        dtype=compute_dtype(),
        quantization_config=quant,
        device_map="auto" if on_gpu else None,
        attn_implementation=model_cfg.get("attn_implementation"),
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        model.eval()
    return model, tokenizer


@torch.no_grad()
def generate(model, tokenizer, conversations: list[list[dict]], max_new_tokens: int = 512) -> list[str]:
    """Greedy-decode a reply for each conversation (list of chat messages)."""
    tokenizer.padding_side = "left"
    texts = [
        tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        for conv in conversations
    ]
    # The chat template already contains <bos>, so don't add special tokens again.
    inputs = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = out[:, inputs["input_ids"].shape[1]:]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
