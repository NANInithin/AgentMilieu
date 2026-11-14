"""Pluggable token counting and best-effort Hub tokenizer discovery."""

import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from huggingface_hub import HfApi, hf_hub_download


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


class _TokenizerUnavailable(Exception):
    """Discovery did not produce a cacheable counter."""


def gguf_token_counter(repo):
    """Best-effort tokenizer.json lookup; never execute remote tokenizer code."""
    try:
        return _hub_counter(repo, os.getenv("HF_TOKEN") or None)
    except _TokenizerUnavailable:
        return None


@lru_cache(maxsize=16)
def _hub_counter(repo, token):
    def load(candidate):
        path = hf_hub_download(repo_id=candidate, filename="tokenizer.json", token=token)
        return HFTokenCounter.from_file(path)

    try:
        return load(repo)
    except Exception:  # noqa: BLE001, S110 -- optional lookup must not block builds or log credentials
        pass
    try:
        card = HfApi(token=token).model_info(repo).card_data
        base = card.get("base_model") if card else None
        candidates = [base] if isinstance(base, str) else base
        for candidate in candidates or []:
            if isinstance(candidate, str):
                try:
                    return load(candidate)
                except Exception:  # noqa: BLE001, S112 -- try the next base without logging credentials
                    continue
    except Exception:  # noqa: BLE001, S110 -- gated/missing/offline metadata uses heuristic counting
        pass
    raise _TokenizerUnavailable
