"""
Strategy 6 — Router: classify each problem by difficulty and route easy
cases to greedy decoding, harder cases to a more expensive strategy
(tree search, per config). Difficulty is estimated cheaply (no extra full
generation) from:
  - verifier confidence on a single greedy pass
  - problem length in tokens
  - (optional) self-consistency agreement, if already computed

Kept as a simple, inspectable threshold rule rather than a trained model —
this is explicitly allowed by the guide ("lightweight classifier"), and a
transparent rule is easier to justify in the paper. Swap in a trained
sklearn classifier here later if the threshold rule proves too coarse.
"""

from src.strategies.greedy import run_greedy
from src.strategies.tree_search import run_tree_search


def estimate_difficulty(item: dict, model, tokenizer, verifier, tokenizer_for_length=None) -> dict:
    """One cheap greedy pass + verifier score as the difficulty signal."""
    greedy_result = run_greedy(model, tokenizer, item)
    confidence = verifier.score(item["question"], greedy_result["generation"]) if verifier else 0.5
    length_tokens = len(tokenizer.encode(item["question"]))
    return {"confidence": confidence, "length_tokens": length_tokens, "greedy_result": greedy_result}


def run_router(model, tokenizer, item: dict, verifier=None, threshold: float = 0.6,
                tree_search_kwargs: dict | None = None) -> dict:
    tree_search_kwargs = tree_search_kwargs or {}
    signal = estimate_difficulty(item, model, tokenizer, verifier)

    if signal["confidence"] >= threshold:
        # Easy: trust the greedy pass we already generated, no extra compute spent.
        result = dict(signal["greedy_result"])
        result["strategy"] = "router->greedy"
    else:
        # Hard: escalate to tree search.
        result = run_tree_search(model, tokenizer, item, verifier=verifier, **tree_search_kwargs)
        result["strategy"] = "router->tree_search"

    result["routing_confidence"] = signal["confidence"]
    return result
