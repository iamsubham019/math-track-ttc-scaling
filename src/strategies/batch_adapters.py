"""
src/strategies/batch_adapters.py

Wraps math's existing, already-debugged per-item strategy functions
(run_greedy, run_best_of_n, run_self_consistency, run_tree_search) with the
code track's batch calling convention:

    run_X_batch(model, tokenizer, num_params, problems, config, ledger) -> list[dict]

This lets src/strategies/router.py and scripts/train_router.py — both
ported from the code track's LatentRouterNet — call every math strategy
through the exact same interface the code track uses, without touching or
re-risking the per-item implementations that already produced this
project's reported results.

best_of_n and tree_search optionally use a verifier for re-ranking/pruning;
since the code track's call sites use a fixed positional signature with no
verifier argument (it doesn't need one — its verifier is free, deterministic
AST+execution, not a trained checkpoint), callers here bind a verifier via
functools.partial BEFORE building the strategy-name -> function dict, so the
call sites themselves stay identical to the code track's.
"""

from shared.flop_utils import FlopRecord
from src.strategies.greedy import run_greedy
from src.strategies.best_of_n import run_best_of_n
from src.strategies.self_consistency import run_self_consistency
from src.strategies.tree_search import run_tree_search


def _log_from_result(ledger, result: dict, strategy: str, item_id, num_params):
    """
    Every math strategy result already carries the token-accounting fields
    src/flop_helper.py needs (input_tokens, output_tokens_total, ...). This
    re-derives a FlopRecord from them for the ledger, which train_router.py
    uses for cost-penalty labeling — kept consistent with, not duplicating,
    the estimated_flops already computed inside each strategy function.
    """
    ledger.log(FlopRecord(
        strategy=strategy,
        problem_id=str(item_id),
        num_params=num_params,
        prompt_tokens=result.get("input_tokens", 0),
        generated_tokens=result.get("output_tokens_total", 0),
        num_generations=1,
        extra_forward_passes=result.get("verifier_calls", 0),
        extra_forward_tokens=0,
    ))


def run_greedy_batch(model, tokenizer, num_params, problems, config, ledger):
    max_new_tokens = config["models"]["max_new_tokens"]
    results = []
    for item in problems:
        r = run_greedy(model, tokenizer, item, max_new_tokens)
        _log_from_result(ledger, r, "greedy", item["id"], num_params)
        results.append(r)
    return results


def run_best_of_n_batch(model, tokenizer, num_params, problems, config, ledger, verifier=None):
    cfg = config["best_of_n"]
    results = []
    for item in problems:
        r = run_best_of_n(
            model, tokenizer, item, n_candidates=cfg["n_candidates"],
            temperature=cfg["temperature"], top_p=cfg["top_p"],
            max_new_tokens=config["models"]["max_new_tokens"], verifier=verifier,
        )
        _log_from_result(ledger, r, "best_of_n", item["id"], num_params)
        results.append(r)
    return results


def run_self_consistency_batch(model, tokenizer, num_params, problems, config, ledger):
    cfg = config["self_consistency"]
    results = []
    for item in problems:
        r = run_self_consistency(
            model, tokenizer, item, n_samples=cfg["n_samples"],
            temperature=cfg["temperature"], top_p=cfg["top_p"],
            max_new_tokens=config["models"]["max_new_tokens"],
        )
        _log_from_result(ledger, r, "self_consistency", item["id"], num_params)
        results.append(r)
    return results


def run_tree_search_batch(model, tokenizer, num_params, problems, config, ledger, verifier=None):
    cfg = config["tree_search"]
    results = []
    for item in problems:
        r = run_tree_search(
            model, tokenizer, item, branching_factor=cfg["branching_factor"],
            max_depth=cfg["max_depth"], beam_width=cfg["beam_width"],
            temperature=cfg["temperature"], verifier=verifier,
        )
        _log_from_result(ledger, r, "tree_search", item["id"], num_params)
        results.append(r)
    return results
