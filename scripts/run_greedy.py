"""
Run greedy decoding baseline over a dataset and model.

Example:
    python scripts/run_greedy.py --model llama3.2-1b --dataset gsm8k --split test --limit 50
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from src.model_utils import load_config, load_model_and_tokenizer
from src.data_utils import load_dataset
from src.strategies.greedy import run_greedy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["llama3.2-1b", "llama3.2-3b"])
    ap.add_argument("--dataset", required=True, choices=["gsm8k", "math"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model, tokenizer = load_model_and_tokenizer(args.model, cfg)
    data = load_dataset(args.dataset, args.split, args.limit)

    os.makedirs(cfg["paths"]["logs_dir"], exist_ok=True)
    out_path = f"{cfg['paths']['logs_dir']}/greedy_{args.model}_{args.dataset}.jsonl"

    n_correct = 0
    with open(out_path, "w") as f:
        for item in tqdm(data, desc=f"greedy/{args.model}/{args.dataset}"):
            result = run_greedy(model, tokenizer, item, cfg["models"]["max_new_tokens"])
            n_correct += int(result["correct"])
            f.write(json.dumps(result) + "\n")

    acc = n_correct / len(data) if data else 0.0
    print(f"\nGreedy | {args.model} | {args.dataset} | acc={acc:.4f} ({n_correct}/{len(data)})")
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
