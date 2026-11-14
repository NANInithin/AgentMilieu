"""Optional, worker-local GGUF comparison backend. No GPU or model imports at startup."""

import ctypes
import math
import os
import re
import site
import threading
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from .retrieval import bm25_search
from .tokens import HeuristicTokenCounter

NEUTRAL_PROMPT = "You are a helpful assistant. Answer clearly and concisely."
_MODELS = {}
_LOCK = threading.Lock()
_CUDA_LIBS = []
_CURATED = [
    "bartowski/Llama-3.2-1B-Instruct-GGUF",
    "bartowski/Llama-3.2-3B-Instruct-GGUF",
    "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
    "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    "bartowski/SmolLM2-1.7B-Instruct-GGUF",
    "bartowski/gemma-2-2b-it-GGUF",
    "bartowski/Phi-3.5-mini-instruct-GGUF",
]


def _api():
    return HfApi(token=os.getenv("HF_TOKEN") or None)


def account_models():
    """Return owner repo IDs; discovery failure is an empty catalog."""
    owner = os.getenv("MILIEU_HF_AUTHOR") or os.getenv("SPACE_AUTHOR_NAME")
    if not owner:
        return []
    try:
        return [model.id for model in _api().list_models(author=owner, filter="gguf")]
    except Exception:  # noqa: BLE001 -- account discovery is deliberately best-effort
        return []


def curated_models():
    return list(_CURATED)


def search_models(query, limit=20):
    if not query.strip():
        return []
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    return [model.id for model in _api().list_models(
        search=query, filter="gguf", sort="downloads", limit=limit)]


def _max_bytes():
    gb = float(os.getenv("MILIEU_MAX_GGUF_GB", "4"))
    if not math.isfinite(gb) or gb <= 0:
        raise ValueError("MILIEU_MAX_GGUF_GB must be positive and finite")
    return int(gb * 1024**3)


def gguf_files(repo):
    """Return {filename, size} entries, preferred Q4_K_M first; omit split models."""
    cap = _max_bytes()
    files = _api().model_info(repo, files_metadata=True).siblings
    result = [{"filename": f.rfilename, "size": f.size} for f in files
              if f.rfilename.lower().endswith(".gguf")
              and "mmproj" not in f.rfilename.lower()
              and not re.search(r"-\d{5}-of-\d{5}", f.rfilename)
              and f.size is not None and 0 < f.size <= cap]
    return sorted(result, key=lambda f: ("q4_k_m" not in f["filename"].lower(),
                                         f["size"], f["filename"]))


def context_prompts(pack, question, *, top_k=3, counter=None):
    """Accept BuildResult, ContextPack, registry payload or exported manifest dict.

    RAG uses the pack's token counter convention (heuristic by default). Whole
    retrieved chunks are kept so untrusted-content labels cannot be truncated.
    """
    counter = counter or HeuristicTokenCounter()
    if hasattr(pack, "pack"):
        pack = pack.pack
    if is_dataclass(pack):
        pack = asdict(pack)
    pack = pack.get("pack", pack)
    prompt = pack["system_prompt"] + "\n\n" + pack["knowledge"]
    if pack["strategy"] == "rag" and pack.get("index"):
        budget = pack["token_accounting"].get("retrieval", 0)
        selected = ""
        for record in bm25_search(pack["index"], pack["corpus"], question, k=top_k):
            addition = f'\n\n[{record["citation"]}]\n{record["text"]}'
            if counter.count(selected + addition) <= budget:
                selected += addition
        prompt += selected
    return prompt, NEUTRAL_PROMPT


def _preload_cuda():
    if _CUDA_LIBS:
        return
    roots = [Path(root) / "nvidia" for root in [*site.getsitepackages(), site.getusersitepackages()]]
    loaded = []
    for pattern in ("cuda_runtime/lib/libcudart.so.12", "cublas/lib/libcublasLt.so.12",
                    "cublas/lib/libcublas.so.12"):
        for root in roots:
            library = root / pattern
            if library.is_file():
                loaded.append(ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL))
                break
    _CUDA_LIBS.extend(loaded)


def _load(path, n_ctx, gpu):
    try:
        _preload_cuda()
        import llama_cpp
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter
    except (ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError("Cannot load llama.cpp. Locally install the chat extra with a CPU wheel; "
                           "on Spaces check the CUDA runtime, cuBLAS and GPU worker driver.") from exc

    use_gpu = gpu and bool(llama_cpp.llama_supports_gpu_offload())
    for device in (["gpu", "cpu"] if use_gpu else ["cpu"]):
        key = (path, n_ctx, device)
        if key in _MODELS:
            return (*_MODELS[key], device)
        try:
            model = llama_cpp.Llama(model_path=path, n_ctx=n_ctx,
                                    n_gpu_layers=-1 if device == "gpu" else 0, verbose=False)
        except Exception:
            if device == "gpu":
                continue
            raise
        template = model.metadata.get("tokenizer.chat_template")
        if not template:
            model.close()
            raise ValueError("GGUF must contain its own tokenizer.chat_template")
        formatter = Jinja2ChatFormatter(
            template=template,
            eos_token=model.detokenize([model.token_eos()], special=True).decode("utf-8"),
            bos_token=model.detokenize([model.token_bos()], special=True).decode("utf-8"))
        model.chat_handler = formatter.to_chat_handler()
        _MODELS[key] = (model, formatter)
        return model, formatter, device


def _messages(model, formatter, system, history, n_ctx, max_tokens):
    messages = [{"role": "system", "content": system}]
    for message in history:
        if message.get("role") not in ("user", "assistant") or not isinstance(
                message.get("content"), str):
            raise ValueError("History must contain text user/assistant messages")
        messages.append({"role": message["role"], "content": message["content"]})
    if len(messages) == 1 or messages[-1]["role"] != "user":
        raise ValueError("History must end with the latest user question")
    while True:
        rendered = formatter(messages=messages)
        count = len(model.tokenize(rendered.prompt.encode("utf-8"),
                                   add_bos=not rendered.added_special, special=True))
        if count + max_tokens <= n_ctx:
            return messages
        if len(messages) <= 2:
            raise ValueError("System prompt and latest question exceed the context window")
        del messages[1]
        while len(messages) > 2 and messages[1]["role"] == "assistant":
            del messages[1]


def compare_chat(repo, filename, *, n_ctx, with_system, without_system=NEUTRAL_PROMPT,
                 with_history, without_history, temperature=0.7, max_tokens=256,
                 top_p=0.95, gpu=True):
    """Generate sequential replies using one worker-local model and chat template.

    Histories are role/content dict lists including the latest user question.
    Returns with_context/without_context reply strings and per-side metadata.
    Call this inside the UI's spaces.GPU worker, never pass a model across workers.
    Oversized static context/questions raise ValueError rather than silently dropping facts.
    """
    if n_ctx <= 0 or not 0 < max_tokens < n_ctx or not 0 <= temperature <= 2 or not 0 < top_p <= 1:
        raise ValueError("Invalid context or sampling parameters")
    if filename not in {f["filename"] for f in gguf_files(repo)}:
        raise ValueError("GGUF file is missing, oversized, split, or a projector")
    path = hf_hub_download(repo_id=repo, filename=filename, token=os.getenv("HF_TOKEN") or None)
    if Path(path).stat().st_size > _max_bytes():
        raise ValueError("Downloaded GGUF exceeds size cap")
    with _LOCK:
        model, formatter, device = _load(path, n_ctx, gpu)
        sides = [("with_context", with_system, with_history),
                 ("without_context", without_system, without_history)]
        prepared = [(side, _messages(model, formatter, system, history, n_ctx, max_tokens))
                    for side, system, history in sides]
        result = {"metadata": {"device": device}}
        for side, messages in prepared:
            start = time.perf_counter()
            completion = model.create_chat_completion(
                messages=messages, temperature=temperature, max_tokens=max_tokens, top_p=top_p)
            result[side] = completion["choices"][0]["message"]["content"] or ""
            result["metadata"][side] = {
                "prompt_tokens": completion["usage"]["prompt_tokens"],
                "generation_seconds": time.perf_counter() - start,
            }
        return result
