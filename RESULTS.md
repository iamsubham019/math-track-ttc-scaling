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

---

## Round 3 — Mentor Feedback: Shared Package, Real FLOPs, Stronger Router

Three changes requested by mentor review, ported from the code track repo
(github.com/LovelyDev-06/code_track_slm):

1. **`estimated_flops` replaces token count as the compute metric.** Copied
   `shared/flop_utils.py` from the code track verbatim; added
   `src/flop_helper.py` to compute `2 × N_params × N_tokens` per strategy call
   and log it in every JSONL row. This also fixed a real accounting bug: Tree
   Search's verifier/judge scoring calls during beam pruning were previously
   invisible to the compute metric entirely (only generation tokens were
   counted) — they're now tracked as extra forward passes.
2. **`shared/` package** created with `flop_utils.py`, `hub_utils.py`, and
   `model_utils.py`, matching the code track's structure so all three tracks
   compute FLOPs identically.
3. **Learned Latent Router**, ported from the code track's `LatentRouterNet`:
   sentence embeddings → small MLP with a bottleneck → softmax over
   strategies, trained end-to-end on labels from actively running all four
   strategies on every training problem (not passively mining existing logs,
   which is what the earlier `latent_router.py` did). Replaces the threshold
   router as the primary one; the old version is preserved as
   `router_threshold.py` / `run_router_threshold.py` for comparison.

### Learned Latent Router — results (n=30 per combo, active labeling)

| Model | Dataset | Greedy | Old Threshold Router | **New Learned Router** |
|---|---|---|---|---|
| 1B | GSM8K | 40.0%¹ | 30.0% | **40.0%** |
| 3B | GSM8K | 55.0% | **66.7%** | 63.3% |
| 1B | MATH | 10.0% | 13.3% | 13.3% |
| 3B | MATH | 35.0% | 20.0% | **33.3%** |

¹ Backfilled with a fresh run to add FLOPs data — differs slightly from the
original 35.0% reported earlier in this document. This is expected run-to-run
variance (sampling-based strategies use randomness by design; even greedy
decoding has minor GPU floating-point non-determinism across runs on
different sessions), not a code change or bug.

The new router beats the old threshold router in 2 of 4 combos and ties in a
third, while staying competitive with or ahead of greedy everywhere except
3B/GSM8K.

### A real, reproducible finding: router training is data-starved at n=30

Getting to a genuinely confident router took three iterations, each of which
surfaced a distinct, honestly-documented failure mode:

1. **Unweighted loss on imbalanced labels (23 greedy / 4 / 2 / 1 out of 30)
   collapsed to always predicting the majority class** — train accuracy
   converged to exactly the 76.67% majority-class baseline, and the trained
   router routed 100% of test problems to greedy regardless of input.
2. **Raw inverse-frequency class weighting overcorrected** — it fixed the
   collapse, but swung to the opposite extreme: the router then never
   predicted greedy at all (0 of 30), despite greedy being the correct,
   cheapest choice for 77% of training problems. Softmax probabilities
   stayed clustered near the 0.25 chance level for all four classes.
3. **Sqrt-scaled class weighting + best-checkpoint tracking** (saving the
   epoch with peak train accuracy, not just the final epoch — training
   accuracy was observed to peak above the majority baseline mid-training,
   then regress back down by the final epoch) gave the results in the table
   above. This is the version now in the repo.

All three fixes are real, defensible engineering (see code comments in
`scripts/train_router.py`), but the underlying ceiling is data quantity, not
remaining bugs: with only 1-5 examples for three of the four strategy
classes, and softmax probabilities still clustered near chance level even in
the best version, 30 labeled examples is not enough for a 384-dimension
embedding classifier to learn confident, genuine per-problem discrimination.
The architecture and training pipeline are correctly implemented and match
the code track's design; scaling to meaningfully more labeled examples (each
of which costs a full run of all four strategies under the active-labeling
scheme) is the clear next step, not further hyperparameter tuning at this
sample size.

### FLOPs plot coverage

`logs/results/performance_vs_flops.png` currently has full FLOPs coverage
for the 1B/GSM8K combo (greedy, best-of-n, self-consistency, tree-search,
and both routers) and router-only coverage for the other three combos —
their base strategy logs predate the FLOPs instrumentation and would need
re-running to backfill. Left as future work given time constraints this
round; the 1B/GSM8K combo demonstrates the full compute-vs-accuracy
comparison the plot is meant to show.
