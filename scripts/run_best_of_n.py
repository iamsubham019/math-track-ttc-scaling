"""
Run Best-of-N sampling over a dataset and model.

Example:
    python scripts/run_best_of_n.py --model llama3.2-1b --dataset gsm8k --limit 50
    # with a trained verifier for re-ranking:
    python scripts/run_best_of_n.py --model llama3.2-1b --dataset gsm8k --verifier checkpoints/verifier
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from src.model_utils import load_config, load_model_and_tokenizer
from src.data_utils import load_dataset
from src.strategies.best_of_n import run_best_of_n
from src.strategies.verifier import Verifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["llama3.2-1b", "llama3.2-3b"])
    ap.add_argument("--dataset", required=True, choices=["gsm8k", "math"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--verifier", default=None, help="path to a trained verifier checkpoint")
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model, tokenizer = load_model_and_tokenizer(args.model, cfg)
    data = load_dataset(args.dataset, args.split, args.limit)
    verifier = Verifier(args.verifier) if args.verifier else None

    os.makedirs(cfg["paths"]["logs_dir"], exist_ok=True)
    out_path = f"{cfg['paths']['logs_dir']}/best_of_n_{args.model}_{args.dataset}.jsonl"

    n_correct = 0
    bn_cfg = cfg["best_of_n"]
    with open(out_path, "w") as f:
        for item in tqdm(data, desc=f"best_of_n/{args.model}/{args.dataset}"):
            result = run_best_of_n(
                model, tokenizer, item,
                n_candidates=bn_cfg["n_candidates"], temperature=bn_cfg["temperature"],
                top_p=bn_cfg["top_p"], max_new_tokens=cfg["models"]["max_new_tokens"],
                verifier=verifier,
            )
            n_correct += int(result["correct"])
            f.write(json.dumps(result) + "\n")

    acc = n_correct / len(data) if data else 0.0
    print(f"\nBest-of-N | {args.model} | {args.dataset} | acc={acc:.4f} ({n_correct}/{len(data)})")
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
