"""
shared/model_utils.py

Generic HF causal-LM loader shared across all three tracks. Track-specific
prompt-building (math's chat-formatted GSM8K/MATH prompt, code's HumanEval/
MBPP wrapper, etc.) stays in each track's own src/data_utils.py or
src/model_utils.py — this file only does model/tokenizer loading, which is
identical everywhere.

Unlike the original src/model_utils.py in this repo (which returns just
(model, tokenizer), to avoid touching every existing call site), this
shared version returns (model, tokenizer, num_params) — new code (the
learned latent router, FLOP-aware strategy adapters) needs num_params for
FLOPs accounting per flop_utils.py's formula, so it's computed once here
instead of separately in every caller.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def resolve_model_name(alias_or_name: str, config: dict) -> str:
    """Accepts either a short alias defined in config['models'] or a full HF repo id."""
    models_cfg = config.get("models", {})
    if alias_or_name in models_cfg:
        return models_cfg[alias_or_name]
    return alias_or_name  # assume it's already a full HF repo id


def load_model_and_tokenizer(alias_or_name: str, config: dict):
    model_name = resolve_model_name(alias_or_name, config)
    dtype = _DTYPE_MAP.get(config["models"].get("dtype", "bfloat16"), torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=config["models"].get("device_map", "auto"),
    )
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    return model, tokenizer, num_params
