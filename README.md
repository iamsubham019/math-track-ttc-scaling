# Math Track — Test-Time Compute Scaling for Small LMs

Partner A deliverable for the "Test-Time Compute Scaling for Small Language Models" project.
Domain: **Math** — Llama-3.2-1B & Llama-3.2-3B on **GSM8K** and **MATH**.

This repo implements the six shared strategies end-to-end and independently of the
Code (Partner B) and Reasoning (Partner C) tracks, per the Team Work Assignment Guide.

**Status: pipeline validated end-to-end; FLOPs-based compute accounting and a**
**learned latent router (ported from the code track) are in.** See
[`RESULTS.md`](RESULTS.md) for the current summary table, findings, and known
limitations (small sample sizes, router training data scarcity) — read that
before citing any numbers in the paper. Full-test-set runs and backfilling
FLOPs across all combos are the next steps, budget permitting.

## Strategies implemented

| # | Strategy | Script |
|---|----------|--------|
| 1 | Greedy Decoding | `scripts/run_greedy.py` |
| 2 | Best-of-N | `scripts/run_best_of_n.py` |
| 3 | Self-Consistency | `scripts/run_self_consistency.py` |
| 4 | Tree Search | `scripts/run_tree_search.py` |
| 5 | Verifier | `scripts/train_verifier.py` (+ used by strategies 2/4/6) |
| 6 | Router | `scripts/run_router.py` |

## Repo layout

```
math-track/
├── configs/config.yaml        # model names, sampling params, paths
├── src/
│   ├── data_utils.py          # GSM8K / MATH loading + prompt formatting
│   ├── answer_utils.py        # answer extraction + correctness checking
│   ├── model_utils.py         # model/tokenizer loading (HF transformers)
│   ├── hub_utils.py           # push checkpoints/results to HF Hub
│   └── strategies/
│       ├── greedy.py
│       ├── best_of_n.py
│       ├── self_consistency.py
│       ├── tree_search.py
│       ├── verifier.py
│       └── router.py
├── scripts/                   # thin CLI entry points, one per strategy
└── logs/, checkpoints/        # created at runtime, gitignored
```

## Quickstart on Kaggle

1. New Kaggle Notebook → enable a GPU accelerator (T4 x2 or P100).
2. Add your Hugging Face token as a Kaggle **Secret** named `HF_TOKEN`
   (needed for gated Llama-3.2 weights and for pushing checkpoints).
3. Clone this repo into the notebook:
   ```bash
   !git clone https://github.com/<your-username>/math-track.git
   %cd math-track
   !pip install -r requirements.txt -q
   ```
4. Log in to HF Hub:
   ```python
   from huggingface_hub import login
   from kaggle_secrets import UserSecretsClient
   login(UserSecretsClient().get_secret("HF_TOKEN"))
   ```
5. Run stage by stage, cheapest first (this also validates your setup early):
   ```bash
   !python scripts/run_greedy.py --model llama3.2-1b --dataset gsm8k --split test --limit 50
   ```
   Drop `--limit` once you've confirmed it runs, and repeat for `llama3.2-3b`, `math`, etc.
6. Each run writes results to `logs/<strategy>_<model>_<dataset>.jsonl` and pushes checkpoints /
   result tables to your HF Hub repo (see `configs/config.yaml`).

## Order of operations (recommended)

1. `run_greedy.py` — cheapest, gives your baseline accuracy + FLOPs-per-problem reference.
2. `run_best_of_n.py` and `run_self_consistency.py` — both just need repeated sampling.
3. `train_verifier.py` — train the correctness/quality scorer; needed for tree search re-ranking
   and for the router's difficulty signal.
4. `run_tree_search.py` — most expensive, do last on a compute budget.
5. `run_router.py` — combines everything: routes each problem to a strategy based on the
   verifier's difficulty signal.
6. `scripts/evaluate_all.py` — aggregates all logs into one results table + a performance-vs-FLOPs
   plot for the paper.

## Compute budgeting knobs

All in `configs/config.yaml`:
- `best_of_n.n_candidates`
- `self_consistency.n_samples`
- `tree_search.branching_factor`, `tree_search.max_depth`
- `models.max_new_tokens`

Start small (`n=4`, `limit=50`) to sanity-check correctness before scaling up to full
GSM8K (1,319 test) / MATH test split — these are what generate your FLOPs-vs-accuracy curve.
