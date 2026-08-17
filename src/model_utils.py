"""
Model + tokenizer loading, shared by every strategy script.
Keeps all HF-specific setup (dtype, device_map, quantization) in one place.
"""

import time
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model_and_tokenizer(model_key: str, cfg: dict):
    """model_key is one of cfg['models'] keys, e.g. 'llama3.2-1b'."""
    model_name = cfg["models"][model_key]
    dtype = getattr(torch, cfg["models"].get("dtype", "bfloat16"))
    load_in_4bit = cfg["models"].get("load_in_4bit", False)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"torch_dtype": dtype, "device_map": "auto"}
    if load_in_4bit:
        kwargs["load_in_4bit"] = True

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, messages: list[dict], max_new_tokens: int = 512,
             temperature: float = 0.0, top_p: float = 1.0, num_return_sequences: int = 1):
    """
    Generates completion(s) for a chat-formatted prompt.

    If `messages` ends with an assistant turn (used by tree_search.py to
    continue partial reasoning), the model must literally extend that text
    rather than start a fresh turn. With add_generation_prompt=True, the
    chat template closes the previous assistant turn and opens a new one —
    causing the model to restart its reasoning from scratch instead of
    continuing. continue_final_message=True avoids that.

    Returns (list[str] generations, dict flop_stats) where flop_stats lets you
    build the performance-vs-FLOPs curve without a separate profiler.
    """
    continuing = len(messages) > 0 and messages[-1]["role"] == "assistant"
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=not continuing,
        continue_final_message=continuing,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    do_sample = temperature > 0.0
    start = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.pad_token_id,
        )
    elapsed = time.time() - start

    generations = []
    output_tokens_total = 0
    for seq in out:
        new_tokens = seq[input_len:]
        output_tokens_total += len(new_tokens)
        generations.append(tokenizer.decode(new_tokens, skip_special_tokens=True))

    # Rough FLOPs proxy: 2 * params * tokens generated (standard transformer inference estimate).
    flop_stats = {
        "input_tokens": input_len,
        "output_tokens_total": output_tokens_total,
        "wall_time_sec": elapsed,
        "num_samples": num_return_sequences,
    }
    return generations, flop_stats