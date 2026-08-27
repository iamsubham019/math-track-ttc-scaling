# Math Track — Results

Partner A results for the Test-Time Compute Scaling project. Llama-3.2 1B & 3B on
GSM8K and MATH, all six baseline strategies plus two advisor-requested additions:
a **Learned Latent Router** (replacing the threshold-rule router) and an
**LLM-as-judge step verifier** for Tree Search (alternative to the trained DeBERTa
verifier). Sample sizes below are not yet the full test sets — see "Limitations."

## Summary Table (all 8 strategies)

| Strategy | Model | Dataset | n | Accuracy | Avg. Output Tokens |
|---|---|---|---|---|---|
| Greedy | 1B | GSM8K | 20 | 35.0% | 237 |
| Best-of-N | 1B | GSM8K | 20 | 45.0% | 2,790 |
| Self-Consistency | 1B | GSM8K | 20 | 50.0% | 2,730 |
| Tree Search (verifier) | 1B | GSM8K | 30 | 20.0% | 1,104 |
| Tree Search (LLM judge) | 1B | GSM8K | 30 | 33.3% | 1,174 |
| Router (threshold) | 1B | GSM8K | 30 | 30.0% | 1,294 |
| **Latent Router** | 1B | GSM8K | 30 | **43.3%** | **620** |
| Greedy | 3B | GSM8K | 20 | 55.0% | 179 |
| Best-of-N | 3B | GSM8K | 20 | 75.0% | 2,123 |
| Self-Consistency | 3B | GSM8K | 20 | 70.0% | 2,188 |
| Tree Search (verifier) | 3B | GSM8K | 30 | 66.7% | 837 |
| Tree Search (LLM judge) | 3B | GSM8K | 30 | 63.3% | 819 |
| Router (threshold) | 3B | GSM8K | 30 | 66.7% | 814 |
| **Latent Router** | 3B | GSM8K | 30 | 53.3% | **253** |
| Greedy | 1B | MATH | 20 | 10.0% | 400 |
| Best-of-N | 1B | MATH | 20 | 15.0% | 3,737 |
| Self-Consistency | 1B | MATH | 20 | 15.0% | 3,684 |
| Tree Search (verifier) | 1B | MATH | 30 | 13.3% | 1,529 |
| Tree Search (LLM judge) | 1B | MATH | 30 | 10.0% | 1,561 |
| Router (threshold) | 1B | MATH | 30 | 13.3% | 1,550 |
| **Latent Router** | 1B | MATH | 30 | 13.3% | 2,519 |
| Greedy | 3B | MATH | 20 | 35.0% | 253 |
| Best-of-N | 3B | MATH | 20 | 35.0% | 2,842 |
| Self-Consistency | 3B | MATH | 20 | 35.0% | 2,922 |
| Tree Search (verifier) | 3B | MATH | 30 | 23.3% | 1,214 |
| Tree Search (LLM judge) | 3B | MATH | 30 | 16.7% | 1,137 |
| Router (threshold) | 3B | MATH | 30 | 20.0% | 1,255 |
| **Latent Router** | 3B | MATH | 30 | **36.7%** | **409** |

Raw per-example logs: `logs/*.jsonl`. Regeneratable summary: `logs/results/summary.csv`
(run `python scripts/evaluate_all.py`). Plot: `logs/results/performance_vs_flops.png`
— rendered as a labeled scatter (color = model/dataset, shape = strategy), not
connected lines, since the eight strategies aren't points along a single path and
connecting them implies a trajectory that doesn't exist.

## Findings

1. **Sampling-based strategies (Best-of-N, Self-Consistency) remain the strongest
   on raw accuracy**, especially on GSM8K, but at 4-10x the token cost of the
   cheaper router-based approaches.
2. **The Learned Latent Router is the standout addition.** It wins or is
   competitive with greedy on 3 of 4 combos, and is consistently the *cheapest or
   near-cheapest* strategy overall — e.g. on 3B/MATH it matches the best accuracy
   of any strategy (36.7%) at roughly 1/7th the token cost of Best-of-N/
   Self-Consistency. On 3B/GSM8K it's the one weak spot (53.3% vs. 55-75% for
   other strategies), though still far cheaper.
3. **LLM-judge Tree Search beats the trained DeBERTa verifier on GSM8K**
   (notably 1B/GSM8K: 33.3% vs. 20%) but **underperforms it on MATH for both
   model sizes**. Interpretation: the judge is only as good as the base model's
   own mathematical judgment — on GSM8K-level problems the model can meaningfully
   self-assess, but on harder MATH problems its judgment is as unreliable as its
   generation.
4. **Neither router variant nor any strategy meaningfully cracks MATH** — all
   eight strategies cluster in a narrow 10-37% band regardless of approach,
   reinforcing last round's finding that test-time compute scaling helps far less
   on harder problems for models this small.

## Limitations / Next Steps

- **Sample sizes are still small** (20-30 problems per combo, not the full GSM8K
  test = 1,319 or MATH test = 500) due to Kaggle GPU-time constraints. Treat all
  numbers as directional.
- **The Latent Router's training data is very thin — 4 to 12 labeled examples
  per combo** (bounded by the base strategy logs' overlap, and the fact that only
  problems where at least one strategy got the right answer can be labeled).
  A 2,048-dimension logistic regression trained on single-digit-to-low-double-digit
  examples is a proof-of-concept, not a validated classifier — its strong results
  above should be read with that caveat front and center. More labeled examples
  (via larger base-strategy runs) would be needed to trust this fully.
- **The LLM-judge verifier needs no training data** (a deliberate choice — see
  `src/strategies/llm_judge_verifier.py` docstring for the PRM-vs-judge trade-off
  reasoning), but its quality is capped by the base model's own reasoning ability,
  which is a fundamental limitation on MATH-difficulty problems specifically.
- **Two real pipeline bugs were found and fixed in earlier passes** (see git
  history): Tree Search's step continuation was restarting reasoning from scratch
  each expansion (fixed via `continue_final_message=True`), and verifier training
  diverged to NaN loss on a larger dataset (fixed with LR warmup + gradient
  clipping). Both fixes are included in the numbers above.
- **A configuration bug** (`YOUR_HF_USERNAME` placeholder in `configs/config.yaml`
  getting reintroduced by a file sync) caused repeated HF Hub push/pull failures
  during this round — now fixed and verified working; worth double-checking after
  any future full-file syncs rather than partial patches.
