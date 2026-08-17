"""
Answer extraction + correctness checking for the math track.
Used by every strategy to score candidates and by evaluate_all.py for final metrics.
"""

import re
from sympy import simplify, sympify
from sympy.parsing.sympy_parser import parse_expr

FINAL_ANSWER_RE = re.compile(r"####\s*(.+?)\s*$", re.MULTILINE)


def extract_answer(generation: str) -> str:
    """Pull the final answer out of a model generation formatted per data_utils.SYSTEM_PROMPT."""
    match = FINAL_ANSWER_RE.search(generation.strip())
    if match:
        return match.group(1).strip()
    # fallback: last number in the text (handles models that forget the '####' marker)
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", generation)
    return numbers[-1].replace(",", "") if numbers else ""


def _normalize(s: str) -> str:
    s = s.strip().rstrip(".")
    s = s.replace(",", "").replace("$", "").replace(" ", "")
    s = s.replace("\\!", "").replace("\\,", "")
    return s


def is_correct(prediction: str, gold: str) -> bool:
    """
    Correctness check with two tiers:
      1. Cheap string/number normalization match (handles GSM8K and most MATH cases).
      2. Symbolic equivalence via sympy as a fallback (handles equivalent but
         differently-formatted expressions, e.g. '1/2' vs '0.5', 'x+1' vs '1+x').
    """
    pred, gold = _normalize(prediction), _normalize(gold)
    if not pred or not gold:
        return False
    if pred == gold:
        return True
    try:
        pred_expr = parse_expr(pred.replace("^", "**"))
        gold_expr = parse_expr(gold.replace("^", "**"))
        return bool(simplify(pred_expr - gold_expr) == 0)
    except Exception:
        return False


def majority_vote(answers: list[str]) -> tuple[str, float]:
    """Self-consistency helper: returns (most common normalized answer, agreement fraction)."""
    if not answers:
        return "", 0.0
    normalized = [_normalize(a) for a in answers]
    counts: dict[str, int] = {}
    for a in normalized:
        counts[a] = counts.get(a, 0) + 1
    best = max(counts, key=counts.get)
    return best, counts[best] / len(normalized)
