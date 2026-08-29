"""
Run the Router (difficulty-based strategy selection: greedy vs tree search).
Requires a trained verifier — train it first with scripts/train_verifier.py.

Example:
    python scripts/run_router.py --model llama3.2-1b --dataset gsm8k \
        --verifier checkpoints/verifier --limit 50
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from src.model_utils import load_config, load_model_and_tokenizer
from src.data_utils import load_dataset
from src.strategies.router import run_router
from src.strategies.verifier import Verifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["llama3.2-1b", "llama3.2-3b"])
    ap.add_argument("--dataset", required=True, choices=["gsm8k", "math"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--verifier", required=True, help="path to a trained verifier checkpoint")
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model, tokenizer = load_model_and_tokenizer(args.model, cfg)
    data = load_dataset(args.dataset, args.split, args.limit)
    verifier = Verifier(args.verifier)

    os.makedirs(cfg["paths"]["logs_dir"], exist_ok=True)
    out_path = f"{cfg['paths']['logs_dir']}/router_{args.model}_{args.dataset}.jsonl"

    r_cfg = cfg["router_threshold"]
    ts_cfg = cfg["tree_search"]
    n_correct = 0
    with open(out_path, "w") as f:
        for item in tqdm(data, desc=f"router/{args.model}/{args.dataset}"):
            result = run_router(
                model, tokenizer, item, verifier=verifier, threshold=r_cfg["threshold"],
                tree_search_kwargs=dict(
                    branching_factor=ts_cfg["branching_factor"], max_depth=ts_cfg["max_depth"],
                    beam_width=ts_cfg["beam_width"], temperature=ts_cfg["temperature"],
                ),
            )
            n_correct += int(result["correct"])
            f.write(json.dumps(result) + "\n")

    acc = n_correct / len(data) if data else 0.0
    print(f"\nRouter | {args.model} | {args.dataset} | acc={acc:.4f} ({n_correct}/{len(data)})")
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
