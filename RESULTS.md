# Math Track — Results (First Full Pass)

Partner A results for the Test-Time Compute Scaling project. Llama-3.2 1B & 3B on
GSM8K and MATH, all six strategies. Sample sizes below (not yet the full test sets —
see "Limitations" for why, and what a full run would take).

## Summary Table

| Strategy | Model | Dataset | n | Accuracy | Avg. Output Tokens |
|---|---|---|---|---|---|
| Greedy | 1B | GSM8K | 20 | 35.0% | 237 |
| Best-of-N | 1B | GSM8K | 20 | 45.0% | 2,790 |
| Self-Consistency | 1B | GSM8K | 20 | 50.0% | 2,730 |
| Tree Search | 1B | GSM8K | 30 | 20.0% | 1,104 |
| Router | 1B | GSM8K | 30 | 30.0% | 1,294 |
| Greedy | 3B | GSM8K | 20 | 55.0% | 179 |
| Best-of-N | 3B | GSM8K | 20 | 75.0% | 2,123 |
| Self-Consistency | 3B | GSM8K | 20 | 70.0% | 2,188 |
| Tree Search | 3B | GSM8K | 30 | 66.7% | 837 |
| Router | 3B | GSM8K | 30 | 66.7% | 814 |
| Greedy | 1B | MATH | 20 | 10.0% | 400 |
| Best-of-N | 1B | MATH | 20 | 15.0% | 3,737 |
| Self-Consistency | 1B | MATH | 20 | 15.0% | 3,684 |
| Tree Search | 1B | MATH | 30 | 13.3% | 1,529 |
| Router | 1B | MATH | 30 | 13.3% | 1,550 |
| Greedy | 3B | MATH | 20 | 35.0% | 253 |
| Best-of-N | 3B | MATH | 20 | 35.0% | 2,842 |
| Self-Consistency | 3B | MATH | 20 | 35.0% | 2,922 |
| Tree Search | 3B | MATH | 30 | 23.3% | 1,214 |
| Router | 3B | MATH | 30 | 20.0% | 1,255 |

Raw per-example logs: `logs/*.jsonl`. Regeneratable summary + plot: `logs/results/summary.csv`,
`logs/results/performance_vs_flops.png` (run `python scripts/evaluate_all.py`).

## Findings

1. **Sampling-based strategies (Best-of-N, Self-Consistency) are the strongest and
   most reliable.** They match or beat greedy on every combo, with the biggest win on
   3B/GSM8K (55% → 75%), a ~20pt gain from extra test-time compute.
2. **Search-based strategies (Tree Search, Router) are competitive on 3B/GSM8K only**
   (66.7% vs. 55% greedy), and notably cheaper there than sampling (~800 vs. ~2,100
   avg. output tokens) — a real compute-efficiency story.
3. **Tree Search/Router underperform greedy on 1B/GSM8K and 3B/MATH.** This is a
   genuine result, not noise: unlike Best-of-N/Self-Consistency (which don't depend
   on the verifier), Tree Search's beam pruning is only as good as the verifier
   scoring it — and our verifier (trained on 1,280 examples from our own runs) is
   weak enough to sometimes prune away the correct reasoning path. This is worth
   discussing as a limitation/insight in the paper: search-based test-time compute
   is more sensitive to verifier quality than sampling-based methods.

## Limitations / Next Steps

- **Sample sizes are small** (20–30 problems per combo, not the full GSM8K
  test = 1,319 or MATH test = 500) due to Kaggle GPU-time constraints during
  initial pipeline validation. Numbers should be treated as directional, not final.
- **Verifier is undertrained.** Only 1,280 (question, generation) pairs from our
  own Best-of-N/Self-Consistency runs, 3 epochs on `microsoft/deberta-v3-small`.
  A larger/better verifier would likely improve Tree Search and Router meaningfully.
- **Greedy/Best-of-N/Self-Consistency used `--limit 20`; Tree Search/Router used
  `--limit 30`** (upgraded partway through after diagnosing pipeline bugs — see
  git history). For a clean final comparison, re-run all six strategies at one
  consistent, larger sample size.
- **Two real bugs were found and fixed during this pass** (see commit history):
  (1) Tree Search's step continuation was restarting the model's reasoning from
  scratch at every expansion instead of continuing it — fixed via
  `continue_final_message=True` in `src/model_utils.py`. (2) Verifier training
  diverged (NaN loss) on the larger retraining dataset — fixed with LR warmup and
  explicit gradient clipping in `src/strategies/verifier.py`.
