"""Shared target selection and summaries for the CLI and session UI."""

import json
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

from .devices import DEVICE_PRESETS, GIB, DeviceProfile, probe_adb
from .models import MODEL_PRESETS, ModelSpec
from .pipeline import SourceInput
from .store import LocalRegistry, NotionStore

FILE_TYPES = {".md", ".txt", ".pdf", ".docx", ".xlsx", ".pptx", ".csv", ".html", ".htm"}


def home():
    load_dotenv()
    return Path(os.getenv("MILIEU_HOME", ".milieu")).expanduser().resolve()


def select_model(preset=None, gguf=None, hf_repo=None, hf_file=None, *, n_layers=None,
                 n_kv_heads=None, head_dim=None, n_ctx_train=None, weights_gb=None,
                 model_name="manual", arch="unknown"):
    manual = (n_layers, n_kv_heads, head_dim, n_ctx_train, weights_gb)
    choices = sum((bool(preset), bool(gguf), bool(hf_repo or hf_file), any(v is not None for v in manual)))
    if choices > 1:
        raise ValueError("Choose one model: preset, local GGUF, HF repo/file, or manual geometry")
    if gguf:
        return ModelSpec.from_gguf(gguf)
    if hf_repo or hf_file:
        if not hf_repo or not hf_file:
            raise ValueError("HF model requires both --hf-repo and --hf-file")
        return ModelSpec.from_hub(hf_repo, hf_file)
    if any(v is not None for v in manual):
        if any(v is None for v in manual):
            raise ValueError("Manual model requires layers, KV heads, head dim, training context and weights GB")
        return ModelSpec(n_layers, n_kv_heads, head_dim, n_ctx_train, int(weights_gb * GIB), model_name, arch)
    try:
        return MODEL_PRESETS[preset or "Llama-3.2-1B"]
    except KeyError as exc:
        raise ValueError("Unknown model preset; run milieu models") from exc


def select_device(preset=None, ram_gb=None, available_gb=None, adb=False):
    if sum((bool(preset), ram_gb is not None or available_gb is not None, adb)) > 1:
        raise ValueError("Choose one device: preset, manual RAM or adb")
    if adb:
        device = probe_adb()
        if device is None:
            raise ValueError("adb unavailable, disconnected or unauthorized; use a device preset")
        return device
    if ram_gb is not None or available_gb is not None:
        if ram_gb is None or available_gb is None:
            raise ValueError("Manual RAM requires both --ram-gb and --available-gb (GiB)")
        return DeviceProfile(int(ram_gb * GIB), int(available_gb * GIB))
    try:
        return DEVICE_PRESETS[preset or "phone-6gb"]
    except KeyError as exc:
        raise ValueError("Unknown device preset; run milieu devices") from exc


def source_inputs(urls=(), names=(), files=()):
    sources = [SourceInput("url", value.strip()) for value in urls if value.strip()]
    sources += [SourceInput("search", value.strip()) for value in names if value.strip()]
    paths = []
    for value in files:
        path = Path(value)
        if path.is_dir():
            paths.extend(p for p in sorted(path.rglob("*")) if p.is_file()
                         and p.suffix.lower() in FILE_TYPES
                         and not any(part.startswith(".") for part in p.relative_to(path).parts))
        elif path.is_file():
            paths.append(path)
        else:
            raise ValueError(f"Source file does not exist: {path.name}")
    sources += [SourceInput("file", str(path.resolve())) for path in dict.fromkeys(paths)]
    if not sources:
        raise ValueError("Provide at least one URL, name/topic or supported file")
    return sources


def status(pack_id, registry):
    entry = registry.get(pack_id)
    return NotionStore().get_status(entry["page_id"]) if entry["backend"] == "notion" else entry["status"]


def safety_rows(pack_id, registry):
    return [[f["kind"], f["severity"], f["action"], f["source_id"], f["chunk_id"], f["rule"]]
            for f in json.loads(registry.load_payload(pack_id)["files"]["safety.json"])]


def ram_rows(budget):
    device, config = budget.device, budget.config
    kv = budget.context_window * budget.kv_bytes_per_token
    weights, compute = budget.model.weights_bytes, budget.compute_buffer_bytes
    index = config.retrieval_overhead_bytes
    other = device.total_ram - budget.usable_ram_bytes
    free = max(0, budget.usable_ram_bytes - weights - compute - kv - index)
    return [[name, round(value / GIB, 3)] for name, value in (
        ("Weights", weights), ("Compute", compute), ("KV for context", kv),
        ("Retrieval/index", index), ("OS / other / safety reserve", other), ("Free usable RAM", free))]


def integrations():
    load_dotenv()
    return {"Context.dev": "configured" if os.getenv("CONTEXT_DEV_API_KEY") else "off (HTTP fallback)",
            "Mistral": "configured" if os.getenv("MISTRAL_API_KEY") else "off (extractive)",
            "Notion": "configured" if os.getenv("NOTION_TOKEN") and os.getenv("NOTION_DATABASE_ID") else "off (local review)",
            "Scorton": "available, opt-in" if shutil.which("node") and shutil.which("npx") else "unavailable (skipped)"}


def registry():
    return LocalRegistry(home())
