"""
Strategy 6 — Router (learned latent router).

Ported from the code track's src/strategies/router.py per mentor feedback:
"The learned latent router should be more strong" — this replaces both the
original threshold-on-verifier-confidence router (preserved for reference
in router_threshold.py) and the hidden-state logistic-regression version
(latent_router.py) with the same architecture the code track uses:

    problem text
       -> frozen sentence embedding (all-MiniLM-L6-v2, 384-dim, CPU-cheap)
       -> Linear + ReLU              (hidden_dim)
       -> Linear                     (latent_dim)   <- the "latent" bottleneck
       -> Linear + softmax           (num_strategies)

The bottleneck forces the network to compress whatever makes a problem
"hard for greedy but fine for tree search" into a learned representation,
rather than hand-picked features. scripts/train_router.py trains this
end-to-end on (embedding, best_strategy_label) pairs, where labels come from
ACTIVELY RUNNING all strategies on a training split (not just mining
whatever logs happen to already exist — the previous latent_router.py's
main weakness, which had as few as 4 labeled examples per combo) and
picking whichever strategy was correct at the lowest FLOP cost.

At inference, run_router() embeds each problem, forwards it through the
trained net, and dispatches to the batch-adapter version of whichever
strategy (greedy / best_of_n / self_consistency / tree_search) the net
picked — reusing math's existing, already-debugged per-item implementations
via src/strategies/batch_adapters.py rather than reimplementing generation.
"""

import torch
import torch.nn as nn
import numpy as np
from sentence_transformers import SentenceTransformer
from safetensors.torch import load_file

from src.strategies.batch_adapters import (
    run_greedy_batch, run_best_of_n_batch, run_self_consistency_batch, run_tree_search_batch,
)


class LatentRouterNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, num_classes: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),   # <- latent bottleneck
        )
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        z = self.encoder(x)          # the "latent" representation
        logits = self.classifier(z)
        return logits, z

    def predict(self, x):
        with torch.no_grad():
            logits, _ = self.forward(x)
            return torch.argmax(logits, dim=-1)


_embedder_cache = {}


def get_embedder(model_name: str) -> SentenceTransformer:
    if model_name not in _embedder_cache:
        _embedder_cache[model_name] = SentenceTransformer(model_name)
    return _embedder_cache[model_name]


def embed_problems(problems, config) -> np.ndarray:
    embedder = get_embedder(config["router"]["embedding_model"])
    texts = [p["question"] for p in problems]  # math's schema uses "question", not code's "prompt"
    return embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def load_router(checkpoint_path: str, config: dict, embedding_dim: int = 384) -> LatentRouterNet:
    strategies = config["router"]["strategies_available"]
    net = LatentRouterNet(
        input_dim=embedding_dim,
        hidden_dim=config["router"]["hidden_dim"],
        latent_dim=config["router"]["latent_dim"],
        num_classes=len(strategies),
    )
    net.load_state_dict(load_file(checkpoint_path))
    net.eval()
    return net


def build_strategy_funcs(verifier=None):
    """
    Binds an optional verifier into best_of_n/tree_search BEFORE building the
    dispatch table, so every entry has the same (model, tokenizer, num_params,
    problems, config, ledger) call signature the code track uses — matching
    the exact call sites in scripts/train_router.py and run_router() below.
    """
    def best_of_n_with_verifier(model, tokenizer, num_params, problems, config, ledger):
        return run_best_of_n_batch(model, tokenizer, num_params, problems, config, ledger, verifier=verifier)

    def tree_search_with_verifier(model, tokenizer, num_params, problems, config, ledger):
        return run_tree_search_batch(model, tokenizer, num_params, problems, config, ledger, verifier=verifier)

    return {
        "greedy": run_greedy_batch,
        "best_of_n": best_of_n_with_verifier,
        "self_consistency": run_self_consistency_batch,
        "tree_search": tree_search_with_verifier,
    }


def run_router(model, tokenizer, num_params, problems, config, ledger, router_net, verifier=None):
    strategies = config["router"]["strategies_available"]
    strategy_funcs = build_strategy_funcs(verifier=verifier)

    embeddings = embed_problems(problems, config)
    x = torch.tensor(embeddings, dtype=torch.float32)
    with torch.no_grad():
        logits, _ = router_net(x)
        probs = torch.softmax(logits, dim=-1)
        predicted_idx = torch.argmax(logits, dim=-1).tolist()
    for problem, idx, prob in zip(problems, predicted_idx, probs.tolist()):
        print(f"Router | {problem['id']} | predicted={idx} | strategy={strategies[idx]} | "
              f"probabilities={[round(v, 4) for v in prob]}")

    results = []
    buckets = {s: [] for s in strategies}
    for problem, idx in zip(problems, predicted_idx):
        buckets[strategies[idx]].append(problem)

    for strategy_name, bucket_problems in buckets.items():
        if not bucket_problems:
            continue
        strategy_fn = strategy_funcs[strategy_name]
        bucket_results = strategy_fn(model, tokenizer, num_params, bucket_problems, config, ledger)
        for r in bucket_results:
            r["strategy"] = "router"
            r["routed_to"] = strategy_name
        results.extend(bucket_results)

    return results
