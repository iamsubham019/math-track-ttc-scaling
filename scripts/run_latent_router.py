"""
Run the Learned Latent Router — predicts a strategy per problem from the
LLM's hidden-state representation, then dispatches to that strategy's
existing implementation. Train it first with scripts/train_latent_router.py.

Example:
    python scripts/run_latent_router.py --model llama3.2-1b --dataset gsm8k \
        --router checkpoints/latent_router_llama3.2-1b_gsm8k.joblib \
        --verifier checkpoints/verifier_v2 --limit 30
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from src.model_utils import load_config, load_model_and_tokenizer
from src.data_utils import load_dataset
from src.strategies.latent_router import LatentRouter
from src.strategies.greedy import run_greedy
from src.strategies.best_of_n import run_best_of_n
from src.strategies.self_consistency import run_self_consistency
from src.strategies.tree_search import run_tree_search
from src.strategies.verifier import Verifier

DISPATCH = {
    "greedy": lambda model, tok, item, cfg, verifier: run_greedy(model, tok, item, cfg["models"]["max_new_tokens"]),
    "best_of_n": lambda model, tok, item, cfg, verifier: run_best_of_n(
        model, tok, item, n_candidates=cfg["best_of_n"]["n_candidates"],
        temperature=cfg["best_of_n"]["temperature"], top_p=cfg["best_of_n"]["top_p"],
        max_new_tokens=cfg["models"]["max_new_tokens"], verifier=verifier),
    "self_consistency": lambda model, tok, item, cfg, verifier: run_self_consistency(
        model, tok, item, n_samples=cfg["self_consistency"]["n_samples"],
        temperature=cfg["self_consistency"]["temperature"], top_p=cfg["self_consistency"]["top_p"],
        max_new_tokens=cfg["models"]["max_new_tokens"]),
    "tree_search": lambda model, tok, item, cfg, verifier: run_tree_search(
        model, tok, item, branching_factor=cfg["tree_search"]["branching_factor"],
        max_depth=cfg["tree_search"]["max_depth"], beam_width=cfg["tree_search"]["beam_width"],
        temperature=cfg["tree_search"]["temperature"], verifier=verifier),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["llama3.2-1b", "llama3.2-3b"])
    ap.add_argument("--dataset", required=True, choices=["gsm8k", "math"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--router", required=True, help="path to a trained latent router (.joblib)")
    ap.add_argument("--verifier", default=None, help="verifier checkpoint, used by best_of_n/tree_search dispatch")
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model, tokenizer = load_model_and_tokenizer(args.model, cfg)
    data = load_dataset(args.dataset, args.split, args.limit)
    latent_router = LatentRouter.load(args.router)
    verifier = Verifier(args.verifier) if args.verifier else None

    os.makedirs(cfg["paths"]["logs_dir"], exist_ok=True)
    out_path = f"{cfg['paths']['logs_dir']}/latent_router_{args.model}_{args.dataset}.jsonl"

    n_correct = 0
    with open(out_path, "w") as f:
        for item in tqdm(data, desc=f"latent_router/{args.model}/{args.dataset}"):
            predicted_strategy = latent_router.predict_strategy(model, tokenizer, item["question"])
            result = DISPATCH[predicted_strategy](model, tokenizer, item, cfg, verifier)
            result["strategy"] = f"latent_router->{predicted_strategy}"
            result["question"] = item["question"]
            n_correct += int(result["correct"])
            f.write(json.dumps(result) + "\n")

    acc = n_correct / len(data) if data else 0.0
    print(f"\nLatent Router | {args.model} | {args.dataset} | acc={acc:.4f} ({n_correct}/{len(data)})")
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
