#!/usr/bin/env python
"""
Train the Learned Latent Router — ported from the code track's
scripts/train_router.py. Unlike the earlier latent_router.py (which passively
mined whatever strategy logs already happened to exist, ending up with as
few as 4-12 labeled examples per combo), this ACTIVELY RUNS every strategy
on each training problem, so every problem gets a complete, dense label
rather than depending on log overlap. Labels are chosen by a cost-penalized
rule: pick the strategy that's correct at the lowest FLOP cost (see
`lam` / cost_penalty_lambda below), not just "any correct strategy."

Progress is checkpointed to JSON after every strategy call (not just every
problem) and pushed to your Hugging Face Hub repo periodically, so a Kaggle
session timeout mid-labeling doesn't lose completed work — restarting this
script resumes exactly where it left off.

Example:
    python scripts/train_router.py --model llama3.2-1b --dataset gsm8k --limit 30
"""
import argparse, os, sys, collections
import torch, torch.nn as nn, yaml
from safetensors.torch import save_file, load_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import load_dataset
from shared.flop_utils import FlopLedger
from shared.model_utils import load_model_and_tokenizer
from src.strategies.router import LatentRouterNet, embed_problems
from src.strategies.batch_adapters import (
    run_greedy_batch, run_best_of_n_batch, run_self_consistency_batch, run_tree_search_batch,
)
from src.strategies.verifier import Verifier
from shared.checkpoint_utils import atomic_json_save, load_json_checkpoint
from shared.hub_utils import push_file, download_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["llama3.2-1b", "llama3.2-3b"])
    p.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "math"])
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--out", default="checkpoints/router.safetensors")
    p.add_argument("--verifier", default=None,
                    help="optional trained DeBERTa verifier checkpoint, used by best_of_n/tree_search "
                         "during labeling (falls back to majority-vote/heuristic if omitted — math has "
                         "no free deterministic verifier the way the code track's AST+exec check does)")
    p.add_argument("--no_push", action="store_true")
    a = p.parse_args()

    with open(a.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    push_every_n = cfg["hub"].get("push_every_n_problems", 10)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    os.makedirs(cfg["paths"]["checkpoints_dir"], exist_ok=True)

    problems = load_dataset(a.dataset, split=a.split, limit=a.limit)
    tag = len(problems)
    progress_path = os.path.join(
        cfg["paths"]["checkpoints_dir"], f"router_labels_{a.model}_{a.dataset}_{a.split}_limit{tag}.json")
    hub_progress = f"checkpoints/{os.path.basename(progress_path)}"

    if not os.path.exists(progress_path) and not a.no_push:
        download_file(cfg, hub_progress, progress_path)
    completed = load_json_checkpoint(progress_path).get("completed", {}) if os.path.exists(progress_path) else {}

    model, tokenizer, num_params = load_model_and_tokenizer(a.model, cfg)
    verifier = Verifier(a.verifier) if a.verifier else None

    strategies = cfg["router"]["strategies_available"]
    lam = cfg["router"]["cost_penalty_lambda"]

    _FUNCS = {
        "greedy": run_greedy_batch,
        "best_of_n": lambda m, t, n, probs, c, l: run_best_of_n_batch(m, t, n, probs, c, l, verifier=verifier),
        "self_consistency": run_self_consistency_batch,
        "tree_search": lambda m, t, n, probs, c, l: run_tree_search_batch(m, t, n, probs, c, l, verifier=verifier),
    }

    for i, problem in enumerate(problems, 1):
        pid = problem["id"]
        completed.setdefault(pid, {})

        for strategy in strategies:
            if strategy in completed[pid]:
                continue

            print(f"[{i}/{len(problems)}] {pid} -> {strategy}")

            l = FlopLedger()
            r = _FUNCS[strategy](model, tokenizer, num_params, [problem], cfg, l)[0]
            fr = l.as_dicts()[0] if l.as_dicts() else {"estimated_flops": r.get("estimated_flops", 0.0)}

            completed[pid][strategy] = {
                "correct": bool(r["correct"]),
                "flops": float(fr.get("estimated_flops", r.get("estimated_flops", 0.0))),
            }

            atomic_json_save({
                "version": 3,
                "completed": completed,
                "metadata": {"model": a.model, "dataset": a.dataset, "split": a.split, "limit": tag},
            }, progress_path)

        if not a.no_push and (i % push_every_n == 0 or i == len(problems)):
            print(f"Pushing progress checkpoint after {i} problems...")
            push_file(progress_path, cfg, hub_progress)

    all_flops = [completed[p["id"]][st]["flops"] for p in problems for st in strategies]
    max_flops = max(all_flops) if all_flops else 1.0
    labels = []
    for problem in problems:
        rec = completed[problem["id"]]
        best = max(strategies, key=lambda st: (1.0 if rec[st]["correct"] else 0.0) - lam * (rec[st]["flops"] / max_flops))
        labels.append(strategies.index(best))

    counts = collections.Counter(labels)
    print("Router label distribution:", {strategies[k]: v for k, v in sorted(counts.items())})

    embeddings = embed_problems(problems, cfg)
    X = torch.tensor(embeddings, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    net = LatentRouterNet(embeddings.shape[1], cfg["router"]["hidden_dim"], cfg["router"]["latent_dim"], len(strategies))
    opt = torch.optim.Adam(net.parameters(), lr=cfg["router"]["lr"])
    loss_fn = nn.CrossEntropyLoss()

    train_state = a.out + ".train.json"
    weight_resume = a.out + ".train.safetensors"
    start_epoch = 0

    if not os.path.exists(train_state) and not a.no_push:
        download_file(cfg, f"checkpoints/{os.path.basename(train_state)}", train_state)
    if not os.path.exists(weight_resume) and not a.no_push:
        download_file(cfg, f"checkpoints/{os.path.basename(weight_resume)}", weight_resume)
    if os.path.exists(train_state) and os.path.exists(weight_resume):
        state = load_json_checkpoint(train_state)
        net.load_state_dict(load_file(weight_resume))
        start_epoch = int(state.get("epoch", 0))
        print(f"Resuming router training from epoch {start_epoch}")

    net.train()
    for epoch in range(start_epoch, cfg["router"]["epochs"]):
        opt.zero_grad()
        logits, _ = net(X)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        acc = (logits.argmax(-1) == y).float().mean().item()
        print(f"epoch {epoch+1}/{cfg['router']['epochs']} loss={loss.item():.4f} train_acc={acc:.2%}")

        save_file(net.state_dict(), weight_resume)
        atomic_json_save({
            "version": 3, "epoch": epoch + 1, "loss": float(loss.item()),
            "metadata": {"model": a.model, "dataset": a.dataset, "split": a.split, "limit": tag},
        }, train_state)

        if not a.no_push:
            push_file(weight_resume, cfg, f"checkpoints/{os.path.basename(weight_resume)}")
            push_file(train_state, cfg, f"checkpoints/{os.path.basename(train_state)}")

    save_file(net.state_dict(), a.out)
    print(f"Saved trained learned latent router to {a.out}")
    if not a.no_push:
        push_file(a.out, cfg, f"checkpoints/{os.path.basename(a.out)}")


if __name__ == "__main__":
    main()
