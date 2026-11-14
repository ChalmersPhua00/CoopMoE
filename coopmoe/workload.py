"""Requests: labeled records from make_requests.py, selected for the partition in
router.box_categories, and the chat formatting that turns a request into token ids.

    {"id": "code-0007", "category": "code", "source": "...", "request": "...", "reference": "..."}
"""
from __future__ import annotations

import random


def select(requests: list[dict], box_categories: list, n: int, box0_share: float = 0.5, seed: int = 0) -> list[dict]:
    """n requests: round(n * box0_share) from box 0's categories and the rest from box 1's, spread
    evenly over each side's categories, in a fixed random order. Categories no box serves are
    left out. The same selection runs under every scheduling mode, so cells compare like for like."""
    by_cat: dict[str, list[dict]] = {}
    for r in requests:
        by_cat.setdefault(r["category"], []).append(r)
    picked, short = [], []
    n0 = int(round(n * box0_share))
    for box, (side, k) in enumerate(zip(box_categories, (n0, n - n0))):
        if not side:
            raise ValueError(f"router.box_categories gives box {box} no categories")
        for i, c in enumerate(side):
            m = k // len(side) + (i < k % len(side))
            have = by_cat.get(c, [])
            if m > len(have):
                short.append(f"{c} needs {m}, the file has {len(have)}")
            picked += have[:m]
    if short:
        raise ValueError("not enough requests: " + "; ".join(short))
    random.Random(seed).shuffle(picked)
    return picked


class Formatter:
    """Turns into token ids with the checkpoint's chat template, and generated ids back into text."""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.eos = tokenizer.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"])

    @classmethod
    def for_store(cls, store_dir: str) -> "Formatter":
        from transformers import AutoTokenizer

        return cls(AutoTokenizer.from_pretrained(store_dir))

    def prompt(self, request: str) -> list[int]:
        """One user message in the chat template, ready for the assistant's reply."""
        return self.tok.apply_chat_template([{"role": "user", "content": request}], add_generation_prompt=True,
                                            tokenize=True, return_dict=False)

    def text(self, ids: list[int]) -> str:
        """Generated ids up to the first end-of-turn token, as text. Generation keeps going past
        it (fixed length); that part counts for throughput but is not the answer."""
        cut = next((i for i, t in enumerate(ids) if t in self.eos), len(ids))
        return self.tok.decode(ids[:cut])

