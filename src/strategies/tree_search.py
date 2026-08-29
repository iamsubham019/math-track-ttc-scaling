"""
Strategy 4 — Tree Search: expand reasoning step by step, score partial paths,
prune weak branches. Implemented as beam search over reasoning steps:

  - At each depth, every surviving path is expanded `branching_factor` ways
    (short, temperature-sampled continuations up to the next step boundary).
  - All candidate continuations are scored (verifier if available, else a
    length-normalized heuristic) and only `beam_width` paths survive per depth.
  - A path terminates early if it contains the '####' final-answer marker.

This keeps the search domain-agnostic at the framework level while the step
boundary ("\\n") and scoring function are the math-specific adaptation.
"""

from src.data_utils import build_prompt, SYSTEM_PROMPT
from src.model_utils import generate
from src.answer_utils import extract_answer, is_correct
from src.flop_helper import estimate_flops

STEP_MAX_TOKENS = 80  # tokens per expansion step
VERIFIER_SCORE_MAX_LEN = 512  # matches Verifier/LLMJudgeVerifier's own truncation, used for FLOP estimate


def _heuristic_score(text: str) -> float:
    """Fallback scorer when no trained verifier is passed: prefers longer,
    more structured partial reasoning (crude proxy for 'more progress made')."""
    return min(len(text.split()) / 40.0, 1.0)


def _expand(model, tokenizer, prefix_messages: list[dict], partial_text: str,
            branching_factor: int, temperature: float):
    """Generate `branching_factor` candidate next steps continuing partial_text."""
    messages = prefix_messages + ([{"role": "assistant", "content": partial_text}] if partial_text else [])
    generations, flop_stats = generate(
        model, tokenizer, messages, max_new_tokens=STEP_MAX_TOKENS,
        temperature=temperature, top_p=0.95, num_return_sequences=branching_factor,
    )
    # Use the full (token-bounded) generation as the step. Truncating at the
    # first newline discards content: models often put a step header on its
    # own line ("## Step 2: ...") with the actual calculation on the next
    # line, so a naive split("\n")[0] keeps only the header and drops the math.
    steps = [g.strip() for g in generations]
    return steps, flop_stats


def run_tree_search(model, tokenizer, item: dict, branching_factor: int = 3,
                     max_depth: int = 4, beam_width: int = 2,
                     temperature: float = 0.7, verifier=None) -> dict:
    prefix_messages = build_prompt(item["question"])
    beams = [""]  # partial reasoning texts
    total_flops = {"input_tokens": 0, "output_tokens_total": 0, "wall_time_sec": 0.0, "num_samples": 0}

    # A verifier/judge call is a real extra forward pass through some model
    # (the DeBERTa classifier, or the LLM judge doing its own generate()) that
    # plain token-counting on the main generation entirely misses — this was
    # the actual bug behind "avg_output_tokens ... is wrong for tree search":
    # tree search's true compute cost includes every scoring call spent
    # pruning the beam, not just the tokens of the surviving path.
    extra_forward_passes = 0
    extra_forward_tokens = 0

    for depth in range(max_depth):
        candidates = []
        for beam_text in beams:
            if "####" in beam_text:
                candidates.append(beam_text)  # already terminated, carry forward unchanged
                continue
            steps, flop_stats = _expand(model, tokenizer, prefix_messages, beam_text,
                                         branching_factor, temperature)
            for k in total_flops:
                total_flops[k] += flop_stats[k]
            for step in steps:
                candidates.append((beam_text + "\n" + step).strip())

        scored = []
        for cand in candidates:
            if verifier:
                score = verifier.score(item["question"], cand)
                extra_forward_passes += 1
                extra_forward_tokens += min(len(cand.split()), VERIFIER_SCORE_MAX_LEN)
            else:
                score = _heuristic_score(cand)  # no model call, no extra FLOP cost
            scored.append((score, cand))
        scored.sort(key=lambda x: x[0], reverse=True)
        beams = [c for _, c in scored[:beam_width]]

        if all("####" in b for b in beams):
            break

    best_text = beams[0]
    prediction = extract_answer(best_text)
    correct = is_correct(prediction, item["gold_answer"])
    flop_fields = estimate_flops(
        model, "tree_search", item["id"], total_flops,
        extra_forward_passes=extra_forward_passes, extra_forward_tokens=extra_forward_tokens,
    )
    return {
        "id": item["id"],
        "question": item["question"],
        "strategy": "tree_search",
        "prediction": prediction,
        "gold": item["gold_answer"],
        "correct": correct,
        "final_reasoning": best_text,
        "depth_reached": depth + 1,
        "beam_width": beam_width,
        "branching_factor": branching_factor,
        "verifier_calls": extra_forward_passes,
        **total_flops,
        **flop_fields,
    }
