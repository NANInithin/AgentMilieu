import struct
import sys
from types import SimpleNamespace

import pytest

from agentmilieu import llm
from agentmilieu.retrieval import make_index


@pytest.fixture
def hub(monkeypatch):
    calls = []

    class API:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def list_models(self, **kwargs):
            calls.append(kwargs)
            return [SimpleNamespace(id="owner/model")]

        def model_info(self, repo, **kwargs):
            return SimpleNamespace(siblings=[
                SimpleNamespace(rfilename=name, size=size) for name, size in [
                    ("small.gguf", 10), ("model-Q4_K_M.gguf", 20),
                    ("huge.gguf", 5 * 1024**3), ("mmproj.gguf", 1),
                    ("model-00001-of-00002.gguf", 1), ("README.md", 1),
                    ("unknown.gguf", None)]])

    monkeypatch.setattr(llm, "HfApi", API)
    monkeypatch.delenv("MILIEU_MAX_GGUF_GB", raising=False)
    return calls


def test_catalog(hub, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setenv("SPACE_AUTHOR_NAME", "space")
    monkeypatch.delenv("MILIEU_HF_AUTHOR", raising=False)
    assert llm.account_models() == ["owner/model"]
    assert hub[-1] == {"author": "space", "filter": "gguf"}
    assert hub[-2] == {"token": "test-token"}
    monkeypatch.setenv("MILIEU_HF_AUTHOR", "override")
    llm.account_models()
    assert hub[-1]["author"] == "override"
    assert llm.search_models("tiny", 4) == ["owner/model"]
    assert hub[-1] == {"search": "tiny", "filter": "gguf", "sort": "downloads", "limit": 4}
    assert llm.search_models(" ") == []
    assert llm.gguf_files("repo") == [{"filename": "model-Q4_K_M.gguf", "size": 20},
                                        {"filename": "small.gguf", "size": 10}]
    monkeypatch.setenv("MILIEU_MAX_GGUF_GB", str(15 / 1024**3))
    assert llm.gguf_files("repo") == [{"filename": "small.gguf", "size": 10}]
    monkeypatch.delenv("SPACE_AUTHOR_NAME")
    monkeypatch.delenv("MILIEU_HF_AUTHOR")
    assert llm.account_models() == ["owner/model"]
    assert hub[-1] == {"author": "NANI-Nithin", "filter": "gguf"}


def test_curated_models():
    expected = ["NANI-Nithin/Qwen3-0.6B-GGUF", "NANI-Nithin/granite-4.2-3b-GGUF"]
    assert llm.curated_models() == expected
    llm.curated_models().clear()
    assert llm.curated_models() == expected


def test_catalog_failure(monkeypatch):
    monkeypatch.setenv("MILIEU_HF_AUTHOR", "owner")
    monkeypatch.setattr(llm, "HfApi", lambda **_: (_ for _ in ()).throw(RuntimeError()))
    assert llm.account_models() == []
    with pytest.raises(ValueError):
        llm.search_models("x", 0)
    monkeypatch.setenv("MILIEU_MAX_GGUF_GB", "nan")
    with pytest.raises(ValueError):
        llm.gguf_files("repo")


@pytest.fixture
def runtime(monkeypatch, tmp_path, hub):
    path = tmp_path / "model.gguf"
    path.write_bytes(b"model")
    monkeypatch.setattr(llm, "hf_hub_download", lambda **_: str(path))
    monkeypatch.setattr(llm, "_MODELS", {})
    loads, requests = [], []

    class Model:
        def __init__(self, **kwargs):
            self.closed = False
            self.context_size = kwargs["n_ctx"]
            self._model = SimpleNamespace(n_ctx_train=lambda: 24)
            self.metadata = {"tokenizer.chat_template": "template"}
            loads.append(kwargs)
            if kwargs["n_gpu_layers"] == -1:
                raise RuntimeError("GPU unavailable")

        def close(self):
            self.closed = True

        def n_ctx(self):
            return self.context_size

        def detokenize(self, *args, **kwargs):
            return b""

        def token_eos(self):
            return 0

        token_bos = token_eos

        def tokenize(self, text, **kwargs):
            return text.split()

        def create_chat_completion(self, **kwargs):
            requests.append(kwargs)
            return {"choices": [{"message": {"content": "reply"}}],
                    "usage": {"prompt_tokens": 9}}

    class Formatter:
        def __init__(self, **kwargs):
            pass

        def __call__(self, messages):
            return SimpleNamespace(prompt=" ".join(m["content"] for m in messages),
                                   added_special=True)

        def to_chat_handler(self):
            return None

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(
        Llama=Model, llama_supports_gpu_offload=lambda: True))
    monkeypatch.setitem(sys.modules, "llama_cpp.llama_chat_format", SimpleNamespace(
        Jinja2ChatFormatter=Formatter))
    return loads, requests


def test_fallback_cache_and_truncation(runtime):
    loads, requests = runtime
    history = [{"role": "user", "content": "old " * 20},
               {"role": "assistant", "content": "old answer"},
               {"role": "user", "content": "latest question"}]
    args = {"n_ctx": 24, "max_tokens": 8, "with_system": "context", "with_history": history,
                "without_history": history}
    first = llm.compare_chat("repo", "small.gguf", **args)
    second = llm.compare_chat("repo", "small.gguf", **args)
    assert first["with_context"] == second["without_context"] == "reply"
    assert first["metadata"]["device"] == "cpu"
    assert first["metadata"]["with_context"]["prompt_tokens"] == 9
    assert len([load for load in loads if load["n_gpu_layers"] == 0]) == 1
    assert all(len(request["messages"]) == 2 for request in requests)
    assert len(history) == 3
    args["with_system"] = "huge " * 30
    with pytest.raises(ValueError, match="exceed"):
        llm.compare_chat("repo", "small.gguf", **args)


def test_inference_errors(runtime):
    args = {"n_ctx": 16, "max_tokens": 8, "with_system": "context", "with_history": [],
                "without_history": []}
    with pytest.raises(ValueError, match="latest"):
        llm.compare_chat("repo", "small.gguf", **args)
    with pytest.raises(ValueError, match="missing"):
        llm.compare_chat("repo", "mmproj.gguf", **args)
    with pytest.raises(ValueError, match="sampling"):
        llm.compare_chat("repo", "small.gguf", **{**args, "max_tokens": 0})


def test_cache_evicts_previous_model(runtime):
    first, _, _ = llm._load("first", 1024, False)
    assert llm._load("first", 1024, False)[0] is first
    assert not first.closed
    second, _, _ = llm._load("second", 1024, False)
    assert first.closed and not second.closed
    third, _, _ = llm._load("second", 2048, False)
    assert second.closed and not third.closed
    assert llm._load("second", 1024, False)[0] is third
    assert not third.closed
    assert len(llm._MODELS) == 1


@pytest.mark.parametrize("reject_at", ["formatter", "completion"])
def test_system_role_fallback(runtime, monkeypatch, reject_at):
    _, requests = runtime
    model_class = sys.modules["llama_cpp"].Llama
    formatter_class = sys.modules["llama_cpp.llama_chat_format"].Jinja2ChatFormatter
    original_completion = model_class.create_chat_completion
    original_formatter = formatter_class.__call__

    def completion(self, **kwargs):
        if reject_at == "completion" and kwargs["messages"][0]["role"] == "system":
            raise ValueError("System role not supported")
        return original_completion(self, **kwargs)

    def formatter(self, messages):
        if reject_at == "formatter" and messages[0]["role"] == "system":
            raise ValueError("System role not supported")
        return original_formatter(self, messages)

    monkeypatch.setattr(model_class, "create_chat_completion", completion)
    monkeypatch.setattr(formatter_class, "__call__", formatter)
    history = [{"role": "user", "content": "old " * 40},
               {"role": "assistant", "content": "old reply"},
               {"role": "user", "content": "latest question"}]
    result = llm.compare_chat("repo", "small.gguf", n_ctx=24, max_tokens=8,
                              with_system="safe context", without_system="neutral assistant",
                              with_history=history, without_history=history)
    assert result["with_context"] == result["without_context"] == "reply"
    assert len(requests) == 2
    assert requests[0]["messages"] == [{"role": "user", "content": "safe context\n\nlatest question"}]
    assert requests[1]["messages"] == [{"role": "user", "content": "neutral assistant\n\nlatest question"}]
    assert len(history) == 3 and history[-1]["content"] == "latest question"


def test_unrelated_completion_error_is_not_retried(runtime, monkeypatch):
    def fail(self, **kwargs):
        raise ValueError("unrelated failure")

    monkeypatch.setattr(sys.modules["llama_cpp"].Llama, "create_chat_completion", fail)
    history = [{"role": "user", "content": "question"}]
    with pytest.raises(ValueError, match="unrelated failure"):
        llm.compare_chat("repo", "small.gguf", n_ctx=24, max_tokens=8,
                         with_system="context", with_history=history, without_history=history)


def test_load_error_reports_architecture_and_both_attempts(runtime, tmp_path, monkeypatch):
    def string(value):
        raw = value.encode()
        return struct.pack("<Q", len(raw)) + raw

    path = tmp_path / "unsupported.gguf"
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 1)
                     + string("general.architecture") + struct.pack("<I", 8)
                     + string("k2-horizon"))
    attempts = []

    def fail(**kwargs):
        attempts.append(kwargs["n_gpu_layers"])
        raise ValueError("Failed to load model from file")

    monkeypatch.setattr(sys.modules["llama_cpp"], "Llama", fail)
    monkeypatch.setattr(sys.modules["llama_cpp"], "__version__", "0.3.19", raising=False)
    with pytest.raises(ValueError, match="k2-horizon") as error:
        llm._load(str(path), 8192, True)
    assert attempts == [-1, 0]
    assert "0.3.19" in str(error.value)
    assert "pick another model" in str(error.value)
    assert "gpu: Failed to load model" in str(error.value)
    assert "cpu: Failed to load model" in str(error.value)


@pytest.mark.parametrize("training,expected,max_tokens", [(4096, 2048, 8), (1500, 1500, 8),
                                                        (4096, 3072, 1200)])
def test_context_grows_to_fit_both_sides(runtime, monkeypatch, training, expected, max_tokens):
    loads, _ = runtime
    cls = sys.modules["llama_cpp"].Llama
    original_init = cls.__init__
    models = []

    def init(self, **kwargs):
        original_init(self, **kwargs)
        self._model = SimpleNamespace(n_ctx_train=lambda: training)
        models.append(self)

    monkeypatch.setattr(cls, "__init__", init)
    history = [{"role": "user", "content": "question"}]
    result = llm.compare_chat("repo", "small.gguf", n_ctx=1024, max_tokens=max_tokens, gpu=False,
                              with_system="fact " * 1200, with_history=history,
                              without_history=history)
    assert result["metadata"]["n_ctx"] == expected
    assert [load["n_ctx"] for load in loads] == [1024, expected]
    assert models[0].closed and not models[1].closed
    assert result["with_context"] == result["without_context"] == "reply"
    repeated = llm.compare_chat("repo", "small.gguf", n_ctx=1024, max_tokens=max_tokens, gpu=False,
                                with_system="fact " * 1200, with_history=history,
                                without_history=history)
    assert repeated["metadata"]["n_ctx"] == expected
    assert [load["n_ctx"] for load in loads] == [1024, expected]
    assert not models[1].closed


def test_context_cannot_grow_past_training(runtime):
    history = [{"role": "user", "content": "question"}]
    with pytest.raises(ValueError, match="training context 24"):
        llm.compare_chat("repo", "small.gguf", n_ctx=16, max_tokens=8, gpu=False,
                         with_system="fact " * 30, with_history=history, without_history=history)


def test_rag_prompt():
    records = [{"chunk_id": "S1.C1", "citation": "S1#c1", "text": "Battery lasts ten hours."},
               {"chunk_id": "S1.C2", "citation": "S1#c2", "text": "UNTRUSTED " * 100}]
    pack = {"system_prompt": "system", "knowledge": "static", "strategy": "rag",
                "token_accounting": {"retrieval": 30}, "corpus": records, "index": make_index(records)}
    prompt, neutral = llm.context_prompts({"pack": pack}, "Battery UNTRUSTED")
    assert "[S1#c1]" in prompt and records[0]["text"] in prompt
    assert "UNTRUSTED" not in prompt
    assert neutral == llm.NEUTRAL_PROMPT
    assert llm.context_prompts(pack, "")[0] == "system\n\nstatic"
    pack["strategy"] = "stuff"
    assert llm.context_prompts(pack, "Battery")[0] == "system\n\nstatic"


def test_cuda_preload(monkeypatch, tmp_path):
    for name in ("cuda_runtime/lib/libcudart.so.12", "cublas/lib/libcublasLt.so.12",
                 "cublas/lib/libcublas.so.12"):
        path = tmp_path / "nvidia" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setattr(llm.site, "getsitepackages", lambda: [str(tmp_path)])
    monkeypatch.setattr(llm.site, "getusersitepackages", lambda: str(tmp_path))
    monkeypatch.setattr(llm, "_CUDA_LIBS", [])
    loaded = []
    monkeypatch.setattr(llm.ctypes, "CDLL", lambda path, mode: loaded.append(path) or object())
    llm._preload_cuda()
    llm._preload_cuda()
    assert len(loaded) == 3
    assert loaded[0].endswith("libcudart.so.12")
    assert loaded[1].endswith("libcublasLt.so.12")


def test_import_error_is_actionable(monkeypatch):
    monkeypatch.setattr(llm, "_preload_cuda", lambda: (_ for _ in ()).throw(OSError("library")))
    with pytest.raises(RuntimeError, match="Cannot load llama.cpp"):
        llm._load("model", 8192, True)
