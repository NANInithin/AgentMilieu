"""Model geometry; sizes and preset geometry are estimates, not compatibility claims."""

import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from gguf import GGUFReader, GGUFValueType


@dataclass(frozen=True)
class ModelSpec:
    n_layers: int
    n_kv_heads: int
    head_dim: int
    n_ctx_train: int
    weights_bytes: int
    name: str = "manual"
    arch: str = "unknown"
    quant: str | None = None
    approximate: bool = False

    def __post_init__(self):
        for key in ("n_layers", "n_kv_heads", "head_dim", "n_ctx_train", "weights_bytes"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{key} must be a positive integer")

    @classmethod
    def from_metadata(cls, metadata: Mapping, weights_bytes: int):
        arch = metadata.get("general.architecture")
        if not isinstance(arch, str) or not arch:
            raise ValueError("GGUF missing general.architecture; use a manual ModelSpec")
        def required(key):
            value = metadata.get(f"{arch}.{key}")
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"GGUF missing/invalid {arch}.{key}; use a manual ModelSpec")
            return value

        heads = metadata.get(f"{arch}.attention.head_count_kv")
        if heads is None:
            heads = required("attention.head_count")
        dim = metadata.get(f"{arch}.attention.key_length")
        if dim is None:
            count = required("attention.head_count")
            embedding = required("embedding_length")
            if embedding % count:
                raise ValueError("GGUF embedding length is not divisible by attention head count")
            dim = embedding // count
        value_dim = metadata.get(f"{arch}.attention.value_length", dim)
        if value_dim != dim:
            raise ValueError("Unequal key/value dimensions require a manual conservative ModelSpec")
        return cls(required("block_count"), heads, dim, required("context_length"),
                   weights_bytes, str(metadata.get("general.name", arch)), arch)

    @classmethod
    def from_gguf(cls, path: str | Path):
        try:
            reader = GGUFReader(str(path))
            metadata = {k: v.contents() for k, v in reader.fields.items()
                        if k.startswith("general.") or not k.startswith("tokenizer.")}
            return cls.from_metadata(metadata, Path(path).stat().st_size)
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"Invalid GGUF metadata: {exc}") from exc

    @classmethod
    def from_hub(cls, repo_id: str, filename: str, *, max_header_bytes: int = 32 << 20):
        """Read metadata using bounded HF ranges, not a full model download.

        Network/auth errors propagate to the caller; offline callers can choose presets/manual.
        """
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        path = f"{repo_id}/{filename}"
        size = fs.info(path)["size"]
        with fs.open(path, "rb", block_size=64 << 10, cache_type="blockcache") as stream:
            metadata = _read_header(stream, max_header_bytes)
        return cls.from_metadata(metadata, size)


def _read_header(stream: BinaryIO, limit: int) -> dict:
    """Parse GGUF v2/v3 metadata without tensor descriptors or tensor payloads."""
    def read(n):
        if n < 0 or stream.tell() + n > limit:
            raise ValueError("GGUF metadata exceeds header byte limit")
        data = stream.read(n)
        if len(data) != n:
            raise ValueError("Truncated GGUF header")
        return data

    if read(4) != b"GGUF":
        raise ValueError("Not a little-endian GGUF file")
    def unpack(fmt):
        return struct.unpack("<" + fmt, read(struct.calcsize("<" + fmt)))[0]

    if unpack("I") not in (2, 3):
        raise ValueError("Unsupported GGUF version (expected v2/v3)")
    unpack("Q")
    count = unpack("Q")
    formats = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
               7: "?", 10: "Q", 11: "q", 12: "d"}

    def value(kind, keep=True, depth=0):
        if depth > 8:
            raise ValueError("GGUF metadata nesting too deep")
        if kind == GGUFValueType.STRING:
            raw = read(unpack("Q"))
            return raw.decode("utf-8") if keep else None
        if kind == GGUFValueType.ARRAY:
            subtype, length = unpack("I"), unpack("Q")
            if length > limit:
                raise ValueError("Invalid GGUF array length")
            for _ in range(length):
                value(subtype, False, depth + 1)
            return None
        if kind not in formats:
            raise ValueError(f"Unsupported GGUF metadata type: {kind}")
        result = unpack(formats[kind])
        if isinstance(result, float) and not math.isfinite(result):
            raise ValueError("Non-finite GGUF metadata")
        return result

    metadata = {}
    for _ in range(count):
        key = value(GGUFValueType.STRING)
        kind = unpack("I")
        keep = not key.startswith("tokenizer.")
        result = value(kind, keep)
        if keep:
            metadata[key] = result
    return metadata


# Q4_K_M file sizes in decimal GB; geometry/context may vary by model revision.
MODEL_PRESETS = {
    name: ModelSpec(layers, heads, dim, ctx, int(gb * 10**9), name, arch, "Q4_K_M", True)
    for name, arch, layers, heads, dim, ctx, gb in [
        ("Llama-3.2-1B", "llama", 16, 8, 64, 131072, 0.81),
        ("Llama-3.2-3B", "llama", 28, 8, 128, 131072, 2.02),
        ("Qwen2.5-0.5B", "qwen2", 24, 2, 64, 32768, 0.40),
        ("Qwen2.5-1.5B", "qwen2", 28, 2, 128, 32768, 0.99),
        ("Qwen2.5-3B", "qwen2", 36, 2, 128, 32768, 1.93),
        ("Gemma-2-2B", "gemma2", 26, 4, 256, 8192, 1.71),
        ("Phi-3.5-mini", "phi3", 32, 32, 96, 131072, 2.39),
        ("SmolLM2-1.7B", "llama", 24, 32, 64, 8192, 1.06),
    ]
}
