"""
Strategy 5 — Verifier: a small classifier that scores (problem, candidate
solution) pairs for correctness/quality. Used to re-rank Best-of-N candidates,
score tree-search branches, and feed the router's difficulty signal.

Training data is built from your own greedy/best-of-n/self-consistency runs:
  - positive examples: (question, generation) where the extracted answer was correct
  - negative examples: (question, generation) where it was incorrect
See scripts/train_verifier.py for the end-to-end training entry point.
"""

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer,
)


class VerifierDataset(Dataset):
    """records: list of {"question": str, "generation": str, "label": int}"""

    def __init__(self, records: list[dict], tokenizer, max_length: int = 512):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        enc = self.tokenizer(
            r["question"], r["generation"],
            truncation=True, max_length=self.max_length, padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(r["label"], dtype=torch.long),
        }


class Verifier:
    """Thin wrapper used at inference time by best_of_n.py / tree_search.py / router.py."""

    def __init__(self, model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def score(self, question: str, generation: str) -> float:
        """Returns P(correct) in [0, 1]."""
        enc = self.tokenizer(question, generation, truncation=True, max_length=512,
                              return_tensors="pt").to(self.device)
        logits = self.model(**enc).logits
        return torch.softmax(logits, dim=-1)[0, 1].item()


def train_verifier(records: list[dict], base_model: str, output_dir: str,
                    epochs: int = 3, batch_size: int = 16, lr: float = 2e-5):
    """records: list of {"question", "generation", "label" (1=correct, 0=incorrect)}"""
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2)

    split = int(0.9 * len(records))
    train_ds = VerifierDataset(records[:split], tokenizer)
    eval_ds = VerifierDataset(records[split:], tokenizer)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=0.1,        # ease into training instead of taking large early steps
        max_grad_norm=1.0,       # explicit gradient clipping (prevents the NaN blowup seen at grad_norm=784)
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=20,
        report_to=[],
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds)
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir
