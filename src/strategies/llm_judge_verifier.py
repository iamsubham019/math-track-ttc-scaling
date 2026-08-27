"""
LLM-as-judge verifier for Tree Search — advisor-requested upgrade over the
DeBERTa sequence classifier for scoring INTERMEDIATE reasoning steps.

Why an LLM judge instead of a trained step-level PRM: a real PRM needs
per-step correct/incorrect labels (was THIS specific step right, not just
the final answer) — we have no such data, and generating it at scale is a
much larger undertaking than anything else in this pipeline. An LLM judge
needs no extra training data: it scores a partial reasoning trace directly
by being prompted to judge it, using the same base model already loaded
for generation (no second model to download/train).

Trade-off to flag honestly: this is weaker than a well-trained PRM would be
(the judge's calibration is whatever the base model's own judgment gives us,
which is a known weakness of small LMs judging their own reasoning), but it
is the tractable option given current data constraints. If step-level labels
become available later (e.g. by hand-annotating a sample of tree search
branches as correct/incorrect), a real PRM (src/strategies/verifier.py's
training scaffolding, retargeted to per-step labels) would be the next step up.
"""

import re
import torch

JUDGE_PROMPT_TEMPLATE = (
    "You are grading a partial, in-progress solution to a math problem. "
    "Judge only whether the reasoning so far is CORRECT and ON TRACK to reach "
    "the right final answer — not whether it is complete.\n\n"
    "Problem: {question}\n\n"
    "Partial reasoning so far:\n{partial}\n\n"
    "Reply with ONLY a single integer from 0 to 100: your confidence (as a "
    "percentage) that this reasoning is correct so far and heading toward the "
    "right answer. Reply with the number only, nothing else."
)


class LLMJudgeVerifier:
    """
    Drop-in replacement for src/strategies/verifier.Verifier — same
    .score(question, generation) -> float interface, so it can be passed
    directly to run_tree_search(..., verifier=judge) with no other changes.

    Reuses the already-loaded generation model/tokenizer by default (no
    second model download), but accepts a different (model, tokenizer) pair
    if you want a separate, possibly stronger, judge model.
    """

    def __init__(self, model, tokenizer, max_new_tokens: int = 8):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def score(self, question: str, generation: str) -> float:
        prompt_text = JUDGE_PROMPT_TEMPLATE.format(question=question.strip(), partial=generation.strip())
        messages = [{"role": "user", "content": prompt_text}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,   # judge deterministically — we want a stable score, not variety
            pad_token_id=self.tokenizer.pad_token_id,
        )
        response = self.tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
        return self._parse_score(response)

    @staticmethod
    def _parse_score(response: str) -> float:
        """Pull the first 0-100 integer out of the judge's reply and normalize to [0, 1]."""
        match = re.search(r"-?\d+", response)
        if not match:
            return 0.5  # judge gave an unparseable reply; neutral fallback rather than crashing the search
        value = max(0, min(100, int(match.group())))
        return value / 100.0
