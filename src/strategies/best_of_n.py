"""
Strategy 2 — Best-of-N: sample N candidates, pick the strongest with a
domain-appropriate correctness check. When a trained verifier is supplied
(src/strategies/verifier.py), it re-ranks candidates instead of relying on
the (test-time-unavailable) gold answer — see `select_with_verifier` below.
Without a verifier, this defaults to majority-vote selection so the script
is runnable standalone before the verifier stage is trained.
"""

from collections import Counter

from src.data_utils import build_prompt
from src.model_utils import generate
from src.answer_utils import extract_answer, is_correct, majority_vote
from src.flop_helper import estimate_flops

VERIFIER_SCORE_MAX_LEN = 512


def run_best_of_n(model, tokenizer, item: dict, n_candidates: int = 8,
                   temperature: float = 0.8, top_p: float = 0.95,
                   max_new_tokens: int = 512, verifier=None) -> dict:
    messages = build_prompt(item["question"])
    generations, flop_stats = generate(
        model, tokenizer, messages, max_new_tokens=max_new_tokens,
        temperature=temperature, top_p=top_p, num_return_sequences=n_candidates,
    )
    candidates = [extract_answer(g) for g in generations]

    extra_forward_passes = 0
    extra_forward_tokens = 0
    if verifier is not None:
        scores = [verifier.score(item["question"], g) for g in generations]
        extra_forward_passes = len(generations)  # one verifier call per candidate — real extra compute
        extra_forward_tokens = sum(min(len(g.split()), VERIFIER_SCORE_MAX_LEN) for g in generations)
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        prediction = candidates[best_idx]
        selection_method = "verifier"
    else:
        prediction, agreement = majority_vote(candidates)
        selection_method = "majority_of_candidates"

    correct = is_correct(prediction, item["gold_answer"])
    flop_fields = estimate_flops(
        model, "best_of_n", item["id"], flop_stats,
        extra_forward_passes=extra_forward_passes, extra_forward_tokens=extra_forward_tokens,
    )
    return {
        "id": item["id"],
        "question": item["question"],
        "strategy": "best_of_n",
        "prediction": prediction,
        "gold": item["gold_answer"],
        "correct": correct,
        "n_candidates": n_candidates,
        "selection_method": selection_method,
        "candidates": candidates,
        "generations": generations,  # full text, needed for verifier training
        **flop_stats,
        **flop_fields,
    }
