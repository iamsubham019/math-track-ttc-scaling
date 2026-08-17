"""
Train the Strategy-5 Verifier from your own best_of_n / self_consistency run logs.
Run those strategies first (they log every candidate generation with correctness),
then train:

    python scripts/train_verifier.py \
        --logs logs/best_of_n_llama3.2-1b_gsm8k.jsonl logs/self_consistency_llama3.2-1b_gsm8k.jsonl \
        --output checkpoints/verifier
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_utils import load_config
from src.answer_utils import is_correct, extract_answer
from src.strategies.verifier import train_verifier


def build_records(log_paths: list[str]) -> list[dict]:
    """
    Turns best_of_n/self_consistency jsonl logs (which store per-candidate
    generations, or for tree_search a single final_reasoning) into
    (question, generation, label) verifier training records.
    """
    records = []
    for path in log_paths:
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                question = row.get("question", row["id"])
                if "candidates" in row and "generations" in row:
                    # best_of_n / self_consistency: score each full candidate generation
                    for cand_answer, gen_text in zip(row["candidates"], row["generations"]):
                        label = int(is_correct(cand_answer, row["gold"]))
                        records.append({"question": question, "generation": gen_text, "label": label})
                elif "generation" in row:
                    label = int(row["correct"])
                    records.append({"question": question, "generation": row["generation"], "label": label})
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True, help="one or more run jsonl logs")
    ap.add_argument("--output", default="checkpoints/verifier")
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    records = build_records(args.logs)
    print(f"Built {len(records)} training records "
          f"({sum(r['label'] for r in records)} positive, {len(records) - sum(r['label'] for r in records)} negative)")

    v_cfg = cfg["verifier"]
    os.makedirs(args.output, exist_ok=True)
    train_verifier(
        records, base_model=v_cfg["base_model"], output_dir=args.output,
        epochs=v_cfg["train_epochs"], batch_size=v_cfg["batch_size"], lr=v_cfg["lr"],
    )
    print(f"Verifier saved to {args.output}")


if __name__ == "__main__":
    main()
