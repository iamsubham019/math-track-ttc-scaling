"""
Learned Latent Router — advisor-requested upgrade over the threshold-on-
verifier-confidence router in src/strategies/router.py.

Instead of a hand-set threshold rule, this trains a small classifier that
takes the base LLM's own hidden-state representation of a problem (its
"latent" — a pooled embedding from an internal layer, not the visible text)
and directly predicts which strategy is likely to solve it best. The
classifier is trained on real outcomes from your existing strategy logs:
for every problem you've already run through Greedy / Best-of-N /
Self-Consistency / Tree Search, we know which of them actually got it right,
so we have real supervision rather than a threshold guess.

Label construction: for each problem, the label is the CHEAPEST strategy
(by avg_output_tokens) that answered it correctly, preferring Greedy over
the rest since running an unnecessary expensive strategy is exactly the
inefficiency a router exists to avoid. If nothing got it right, the label
is the strategy that came closest isn't knowable from correctness alone, so
such examples are dropped from training (we only have signal on what worked).
"""

import json
import glob
import os
from collections import defaultdict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

# cheapest-first tie-break order when multiple strategies get a problem right
STRATEGY_PRIORITY = ["greedy", "tree_search", "best_of_n", "self_consistency"]


def extract_hidden_state(model, tokenizer, question: str) -> np.ndarray:
    """
    Pools the base LLM's last-layer hidden states over the question text into
    a single fixed-size vector — the "latent" the router classifies on.
    Mean-pooling over tokens (rather than just the last token) is more
    stable for short problem statements like GSM8K/MATH questions.
    """
    inputs = tokenizer(question, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    last_hidden = out.hidden_states[-1]                    # (1, seq_len, hidden_dim)
    pooled = last_hidden.mean(dim=1).squeeze(0)             # (hidden_dim,)
    return pooled.float().cpu().numpy()


def _load_log(path: str) -> dict:
    """Returns {id: {"correct": bool, "output_tokens_total": int}}"""
    out = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            out[row["id"]] = {
                "correct": row["correct"],
                "output_tokens_total": row.get("output_tokens_total", 0),
                "question": row.get("question"),
            }
    return out


def build_training_data(logs_dir: str, model_key: str, dataset: str):
    """
    Scans logs_dir for greedy/best_of_n/self_consistency/tree_search logs
    matching model_key + dataset, and returns (questions, labels) — the
    question text and the cheapest-correct strategy name per problem id.
    """
    per_strategy = {}
    for strategy in STRATEGY_PRIORITY:
        path = f"{logs_dir}/{strategy}_{model_key}_{dataset}.jsonl"
        if os.path.exists(path):
            per_strategy[strategy] = _load_log(path)

    if not per_strategy:
        raise FileNotFoundError(
            f"No strategy logs found for {model_key}/{dataset} in {logs_dir}. "
            "Run greedy/best_of_n/self_consistency/tree_search first."
        )

    # union of all problem ids seen across strategies
    all_ids = set()
    for records in per_strategy.values():
        all_ids.update(records.keys())

    questions, labels = [], []
    for pid in sorted(all_ids):
        best_strategy = None
        for strategy in STRATEGY_PRIORITY:  # already cheapest-first
            rec = per_strategy.get(strategy, {}).get(pid)
            if rec and rec["correct"]:
                best_strategy = strategy
                question = rec.get("question")
                break
        if best_strategy is None:
            continue  # no strategy solved it — no positive supervision available, skip

        if question is None:
            continue  # older log without the "question" field — skip rather than guess
        questions.append(question)
        labels.append(best_strategy)

    return questions, labels


class LatentRouter:
    """Inference-time wrapper: extract hidden state -> predict strategy name."""

    def __init__(self, classifier: LogisticRegression, scaler: StandardScaler, classes: list[str]):
        self.classifier = classifier
        self.scaler = scaler
        self.classes = classes

    def predict_strategy(self, model, tokenizer, question: str) -> str:
        feat = extract_hidden_state(model, tokenizer, question).reshape(1, -1)
        feat = self.scaler.transform(feat)
        return self.classifier.predict(feat)[0]

    def predict_proba(self, model, tokenizer, question: str) -> dict:
        feat = extract_hidden_state(model, tokenizer, question).reshape(1, -1)
        feat = self.scaler.transform(feat)
        probs = self.classifier.predict_proba(feat)[0]
        return dict(zip(self.classifier.classes_, probs))

    def save(self, path: str):
        joblib.dump({"classifier": self.classifier, "scaler": self.scaler, "classes": self.classes}, path)

    @classmethod
    def load(cls, path: str) -> "LatentRouter":
        data = joblib.load(path)
        return cls(data["classifier"], data["scaler"], data["classes"])


def train_latent_router(model, tokenizer, questions: list[str], labels: list[str]) -> LatentRouter:
    """
    Extracts hidden-state features for every training question and fits a
    multinomial logistic regression classifier. Logistic regression (not a
    deep MLP) is a deliberate choice: with only a few hundred labeled
    examples, a linear classifier on top of the LLM's own rich hidden-state
    features is far less prone to overfitting than a deeper head would be.
    """
    features = np.stack([extract_hidden_state(model, tokenizer, q) for q in questions])

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    classifier = LogisticRegression(max_iter=1000, class_weight="balanced", multi_class="auto")
    classifier.fit(features_scaled, labels)

    return LatentRouter(classifier, scaler, sorted(set(labels)))
