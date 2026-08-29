"""Strategy 1 — Greedy Decoding: single best-token generation, the baseline."""

from src.data_utils import build_prompt
from src.model_utils import generate
from src.answer_utils import extract_answer, is_correct
from src.flop_helper import estimate_flops


def run_greedy(model, tokenizer, item: dict, max_new_tokens: int = 512) -> dict:
    messages = build_prompt(item["question"])
    generations, flop_stats = generate(
        model, tokenizer, messages, max_new_tokens=max_new_tokens, temperature=0.0
    )
    prediction = extract_answer(generations[0])
    correct = is_correct(prediction, item["gold_answer"])
    flop_fields = estimate_flops(model, "greedy", item["id"], flop_stats)
    return {
        "id": item["id"],
        "question": item["question"],
        "strategy": "greedy",
        "prediction": prediction,
        "gold": item["gold_answer"],
        "correct": correct,
        "generation": generations[0],
        **flop_stats,
        **flop_fields,
    }
