"""Ephemeral, untrusted input records. Offsets address extracted source text, not file bytes."""

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from .tokens import HeuristicTokenCounter


@dataclass(frozen=True)
class Provenance:
    source_id: str
    source_type: Literal["url", "search", "file"]
    uri: str
    title: str
    fetched_at: datetime
    content_hash: str
    fallback: bool


@dataclass(frozen=True)
class Document:
    source_id: str
    source_type: Literal["url", "search", "file"]
    uri: str
    title: str
    text: str = field(repr=False)
    fetched_at: datetime
    content_hash: str
    fallback: bool = False

    @classmethod
    def create(cls, text: str, *, source_type="file", uri="", title="", fallback=False):
        identity = f"{source_type}:{uri}".encode()
        return cls("S" + hashlib.sha256(identity).hexdigest()[:12], source_type, uri, title,
                   text, datetime.now(timezone.utc), hashlib.sha256(text.encode()).hexdigest(),
                   fallback)


@dataclass(frozen=True)
class Chunk:
    source_id: str
    chunk_id: str
    text: str = field(repr=False)
    start: int
    end: int


def chunk_document(document: Document, *, size: int = 512, overlap: int = 64,
                   counter=None) -> list[Chunk]:
    """Token-limited chunks, preferring Markdown headings then line/word boundaries.

    Overlap is a token ceiling; chunks retain exact source substrings including whitespace.
    """
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Require size > 0 and 0 <= overlap < size")
    counter = counter or HeuristicTokenCounter()
    text = document.text
    headings = [match.start() for match in re.finditer(r"(?m)^#{1,6}\s+", text)]
    chunks = []
    start = 0
    while start < len(text):
        lo, hi = start + 1, len(text)
        end = start
        while lo <= hi:
            mid = (lo + hi) // 2
            if counter.count(text[start:mid]) <= size:
                end, lo = mid, mid + 1
            else:
                hi = mid - 1
        if end == start:
            raise ValueError("Token size too small for one source character")
        heading_boundary = False
        if end < len(text):
            candidates = [h for h in headings if start < h <= end]
            if candidates:
                end = candidates[-1]
                heading_boundary = True
            else:
                boundary = max(text.rfind("\n", start + 1, end), text.rfind(" ", start + 1, end))
                if boundary > start:
                    end = boundary + 1
        while end > start and counter.count(text[start:end]) > size:
            end -= 1
            heading_boundary = False
        if end == start:
            raise ValueError("Token size too small for one source character")
        chunks.append(Chunk(document.source_id, f"{document.source_id}.C{len(chunks) + 1}",
                            text[start:end], start, end))
        if end == len(text):
            break
        next_start = end
        while (not heading_boundary and next_start > start + 1
               and counter.count(text[next_start - 1:end]) <= overlap):
            next_start -= 1
        start = max(start + 1, next_start)
    return chunks
