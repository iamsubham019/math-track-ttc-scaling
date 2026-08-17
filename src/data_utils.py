"""
Dataset loading + prompt formatting for GSM8K and MATH.
Both datasets are exposed through a single common interface:

    load_dataset(name, split, limit=None) -> list[dict]
        each dict has: {"id": str, "question": str, "gold_answer": str}

    build_prompt(question) -> str
        chat-formatted prompt asking for step-by-step reasoning ending in
        a clearly delimited final answer (used by every strategy so that
        answer_utils.extract_answer() can parse it consistently).
"""

import re
from datasets import load_dataset as hf_load_dataset

SYSTEM_PROMPT = (
    "You are a careful math problem solver. Work through the problem step by "
    "step, then give the final answer on its own line in the exact form:\n"
    "#### <answer>\n"
    "Only the final numeric or symbolic answer should follow '####'."
)


def build_prompt(question: str) -> list[dict]:
    """Returns a chat-format message list, ready for tokenizer.apply_chat_template."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]


def _load_gsm8k(split: str, limit: int | None):
    ds = hf_load_dataset("openai/gsm8k", "main", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    items = []
    for i, row in enumerate(ds):
        # GSM8K gold answers are the text after the final "####" in `answer`
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        items.append({"id": f"gsm8k_{split}_{i}", "question": row["question"], "gold_answer": gold})
    return items


def _load_math(split: str, limit: int | None):
    ds = hf_load_dataset("hendrycks/competition_math", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    items = []
    for i, row in enumerate(ds):
        gold = _extract_boxed(row["solution"]) or row["solution"].strip()
        items.append({
            "id": f"math_{split}_{i}",
            "question": row["problem"],
            "gold_answer": gold,
            "level": row.get("level"),
            "type": row.get("type"),
        })
    return items


def _extract_boxed(solution_text: str) -> str | None:
    """MATH gold answers are wrapped in \\boxed{...} inside the solution."""
    idx = solution_text.rfind("\\boxed")
    if idx == -1:
        return None
    i = solution_text.find("{", idx)
    if i == -1:
        return None
    depth, j = 1, i + 1
    while j < len(solution_text) and depth > 0:
        if solution_text[j] == "{":
            depth += 1
        elif solution_text[j] == "}":
            depth -= 1
        j += 1
    return solution_text[i + 1 : j - 1].strip()


LOADERS = {"gsm8k": _load_gsm8k, "math": _load_math}


def load_dataset(name: str, split: str = "test", limit: int | None = None):
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(LOADERS)}.")
    return LOADERS[name](split, limit)
