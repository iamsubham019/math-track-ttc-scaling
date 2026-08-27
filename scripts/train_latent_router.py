"""
Train the Learned Latent Router from your existing strategy logs.
Run greedy/best_of_n/self_consistency/tree_search first (you already have —
this reuses those logs directly, no new generation needed to build labels).

Example:
    python scripts/train_latent_router.py --model llama3.2-1b --dataset gsm8k
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_utils import load_config, load_model_and_tokenizer
from src.strategies.latent_router import build_training_data, train_latent_router


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["llama3.2-1b", "llama3.2-3b"])
    ap.add_argument("--dataset", required=True, choices=["gsm8k", "math"])
    ap.add_argument("--output", default=None, help="defaults to checkpoints/latent_router_<model>_<dataset>.joblib")
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    output = args.output or f"checkpoints/latent_router_{args.model}_{args.dataset}.joblib"
    os.makedirs(os.path.dirname(output), exist_ok=True)

    questions, labels = build_training_data(cfg["paths"]["logs_dir"], args.model, args.dataset)
    print(f"Built {len(questions)} labeled examples.")
    from collections import Counter
    print("Label distribution:", dict(Counter(labels)))

    if len(questions) < 10:
        print("WARNING: very few labeled examples — router will likely be unreliable. "
              "Consider running more of the base strategies first to get more coverage.")

    model, tokenizer = load_model_and_tokenizer(args.model, cfg)
    router = train_latent_router(model, tokenizer, questions, labels)
    router.save(output)
    print(f"Latent router saved to {output}")


if __name__ == "__main__":
    main()
