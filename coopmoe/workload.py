"""The chat formatting that turns a request (a record from make_requests.py) into token ids.

    {"id": "code-0007", "category": "code", "box": 0, "source": "...", "request": "...", "reference": "..."}
"""
from __future__ import annotations


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

