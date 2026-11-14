"""Pluggable token counters; no model/tokenizer downloads."""

import math
from pathlib import Path
from typing import Protocol


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class HeuristicTokenCounter:
    """Approximately four Unicode characters/token; inaccurate for code and non-English text."""

    def count(self, text: str) -> int:
        return math.ceil(len(text) / 4)


class HFTokenCounter:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_file(cls, path: str | Path):
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError("Install agentmilieu[tokenizers] for HF token counting") from exc
        return cls(Tokenizer.from_file(str(path)))

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)
