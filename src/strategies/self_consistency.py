"""Strategy 3 — Self-Consistency: sample N reasoning chains, majority-vote the final answer."""

from src.data_utils import build_prompt
from src.model_utils import generate
from src.answer_utils import extract_answer, is_correct, majority_vote
from src.flop_helper import estimate_flops


def run_self_consistency(model, tokenizer, item: dict, n_samples: int = 8,
                          temperature: float = 0.7, top_p: float = 0.95,
                          max_new_tokens: int = 512) -> dict:
    messages = build_prompt(item["question"])
    generations, flop_stats = generate(
        model, tokenizer, messages, max_new_tokens=max_new_tokens,
        temperature=temperature, top_p=top_p, num_return_sequences=n_samples,
    )
    candidates = [extract_answer(g) for g in generations]
    prediction, agreement = majority_vote(candidates)
    correct = is_correct(prediction, item["gold_answer"])
    flop_fields = estimate_flops(model, "self_consistency", item["id"], flop_stats)
    return {
        "id": item["id"],
        "question": item["question"],
        "strategy": "self_consistency",
        "prediction": prediction,
        "gold": item["gold_answer"],
        "correct": correct,
        "agreement": agreement,          # useful router signal
        "n_samples": n_samples,
        "candidates": candidates,
        "generations": generations,      # full text, needed for verifier training
        **flop_stats,
        **flop_fields,
    }
