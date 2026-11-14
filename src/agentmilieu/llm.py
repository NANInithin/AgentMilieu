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

from .models import _read_header
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
    devices = ["gpu", "cpu"] if use_gpu else ["cpu"]
    failures = []
    for device in devices:
        for (cached_path, cached_ctx, cached_device), cached in _MODELS.items():
            if cached_path == path and cached_device == device and cached_ctx >= n_ctx:
                return (*cached, device)
    while _MODELS:
        _, (previous, _) = _MODELS.popitem()
        previous.close()
    for device in devices:
        key = (path, n_ctx, device)
        try:
            model = llama_cpp.Llama(model_path=path, n_ctx=n_ctx,
                                    n_gpu_layers=-1 if device == "gpu" else 0, verbose=False)
        except Exception as exc:
            failures.append(f"{device}: {exc}")
            if device == "gpu":
                continue
            architecture = "unknown"
            try:
                with open(path, "rb") as stream:
                    architecture = _read_header(stream, 32 << 20).get("general.architecture", "unknown")
            except (OSError, ValueError, UnicodeError):
                pass
            version = getattr(llama_cpp, "__version__", "unknown")
            raise ValueError(
                f"Cannot load model architecture '{architecture}' with the bundled llama.cpp "
                f"(llama-cpp-python {version}). The architecture may be unsupported; pick another "
                f"model or a compatible llama.cpp build. Original load errors: {'; '.join(failures)}"
            ) from exc
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


def _system_role_error(exc):
    return bool(re.search(r"system (?:role|messages?).*(?:not supported|unsupported)",
                          str(exc), re.IGNORECASE))


def _merge_system(messages):
    merged = [dict(message) for message in messages[1:]]
    first_user = next(message for message in merged if message["role"] == "user")
    first_user["content"] = messages[0]["content"] + "\n\n" + first_user["content"]
    return merged


class _ContextOverflow(ValueError):
    def __init__(self, required):
        super().__init__("System prompt and latest question exceed the context window "
                         f"(need {required} tokens including generation)")
        self.required = required


def _messages(model, formatter, system, history, n_ctx, max_tokens, *, merge_system=False):
    messages = [{"role": "system", "content": system}]
    for message in history:
        if message.get("role") not in ("user", "assistant") or not isinstance(
                message.get("content"), str):
            raise ValueError("History must contain text user/assistant messages")
        messages.append({"role": message["role"], "content": message["content"]})
    if len(messages) == 1 or messages[-1]["role"] != "user":
        raise ValueError("History must end with the latest user question")
    while True:
        candidate = _merge_system(messages) if merge_system else messages
        try:
            rendered = formatter(messages=candidate)
        except ValueError as exc:
            if merge_system or not _system_role_error(exc):
                raise
            merge_system = True
            continue
        count = len(model.tokenize(rendered.prompt.encode("utf-8"),
                                   add_bos=not rendered.added_special, special=True))
        if count + max_tokens <= n_ctx:
            return candidate
        if len(messages) <= 2:
            raise _ContextOverflow(count + max_tokens)
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
    Oversized prompts grow the context up to training limits rather than dropping pack facts.
    """
    if n_ctx <= 0 or max_tokens <= 0 or not 0 <= temperature <= 2 or not 0 < top_p <= 1:
        raise ValueError("Invalid context or sampling parameters")
    if filename not in {f["filename"] for f in gguf_files(repo)}:
        raise ValueError("GGUF file is missing, oversized, split, or a projector")
    path = hf_hub_download(repo_id=repo, filename=filename, token=os.getenv("HF_TOKEN") or None)
    if Path(path).stat().st_size > _max_bytes():
        raise ValueError("Downloaded GGUF exceeds size cap")
    with _LOCK:
        sides = [("with_context", with_system, with_history),
                 ("without_context", without_system, without_history)]
        while True:
            model, formatter, device = _load(path, n_ctx, gpu)
            n_ctx = model.n_ctx()
            try:
                return _generate_pair(model, formatter, sides, n_ctx, device,
                                      temperature, max_tokens, top_p)
            except _ContextOverflow as exc:
                training = model._model.n_ctx_train()
                grown = min(training, math.ceil(exc.required / 1024) * 1024)
                if grown <= n_ctx:
                    raise ValueError(f"Prompt needs {exc.required} tokens including generation, "
                                     f"which exceeds model training context {training}. Reduce the pack, "
                                     "question or max new tokens, or choose another model.") from exc
                n_ctx = grown


def _generate_pair(model, formatter, sides, n_ctx, device, temperature, max_tokens, top_p):
    prepared = [(side, system, history,
                 _messages(model, formatter, system, history, n_ctx, max_tokens))
                for side, system, history in sides]
    result = {"metadata": {"device": device, "n_ctx": n_ctx}}
    for side, system, history, messages in prepared:
        start = time.perf_counter()
        try:
            completion = model.create_chat_completion(
                messages=messages, temperature=temperature, max_tokens=max_tokens, top_p=top_p)
        except ValueError as exc:
            if not _system_role_error(exc) or messages[0]["role"] != "system":
                raise
            messages = _messages(model, formatter, system, history, n_ctx, max_tokens,
                                 merge_system=True)
            completion = model.create_chat_completion(
                messages=messages, temperature=temperature, max_tokens=max_tokens, top_p=top_p)
        result[side] = completion["choices"][0]["message"]["content"] or ""
        result["metadata"][side] = {
            "prompt_tokens": completion["usage"]["prompt_tokens"],
            "generation_seconds": time.perf_counter() - start,
        }
    return result
