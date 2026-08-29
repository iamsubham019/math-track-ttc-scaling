"""
src/flop_helper.py

Thin math-track glue around shared/flop_utils.py. Each strategy already
returns token-accounting fields (input_tokens, output_tokens_total,
num_samples) from src/model_utils.generate()'s flop_stats — this module
turns those into an estimated_flops figure using the standard
2 * N_params * N_tokens inference approximation (see shared/flop_utils.py
for the exact formula and reasoning), so every strategy's JSONL output
carries the same estimated_flops field the other two tracks use.

num_params is cached per model object (id()-keyed) since it doesn't change
across calls and recomputing sum(p.numel() for p in model.parameters())
on every single problem, while cheap, is needless repeated work.
"""

from shared.flop_utils import FlopRecord

_num_params_cache = {}


def get_num_params(model) -> int:
    key = id(model)
    if key not in _num_params_cache:
        _num_params_cache[key] = sum(p.numel() for p in model.parameters())
    return _num_params_cache[key]


def estimate_flops(model, strategy: str, problem_id: str, flop_stats: dict,
                    extra_forward_passes: int = 0, extra_forward_tokens: int = 0) -> dict:
    """
    flop_stats is the dict already returned by src/model_utils.generate():
    {"input_tokens", "output_tokens_total", "wall_time_sec", "num_samples"}.
    Returns a dict of fields to merge into a strategy's result dict —
    includes estimated_flops plus the underlying FlopRecord breakdown so
    evaluate_all.py and any downstream analysis can inspect where compute
    went (prompt re-encoding vs. generation vs. extra judge/verifier calls).
    """
    record = FlopRecord(
        strategy=strategy,
        problem_id=problem_id,
        num_params=get_num_params(model),
        prompt_tokens=flop_stats.get("input_tokens", 0),
        generated_tokens=flop_stats.get("output_tokens_total", 0),
        num_generations=1,  # token counts above are already totals across samples where relevant
        extra_forward_passes=extra_forward_passes,
        extra_forward_tokens=extra_forward_tokens,
    )
    d = record.to_dict()
    # avoid clobbering the strategy's own "strategy"/"problem_id"-equivalent keys
    return {"estimated_flops": d["estimated_flops"], "flop_num_params": d["num_params"]}
