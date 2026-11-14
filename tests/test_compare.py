import importlib.util
import sys
from asyncio import BaseEventLoop
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import gradio as gr
import pytest

from agentmilieu import gpu, llm, web


@pytest.fixture
def session(monkeypatch, tmp_path):
    monkeypatch.setenv("MILIEU_HOME", str(tmp_path))
    monkeypatch.setattr(web.ui, "load_dotenv", lambda: None)
    monkeypatch.setattr(web, "_training_context", lambda *_: 8192)
    return web.build_handler(files=["examples/specs.md", "examples/untrusted.md"])[0]


def test_compare_setup_empty_and_built(session):
    empty = web.compare_setup(None)
    assert "Build a pack" in empty[0]
    assert empty[1]["interactive"] is False
    ready = web.compare_setup(session)
    assert "Pending review" in ready[0] and "draft" in ready[0]
    assert ready[1]["interactive"] is True
    assert ready[2]["value"] == web._session_pack(session)["budget"]["context_window"]
    assert "Llama-3.2-1B" in ready[3]["value"]
    assert ready[5]["value"] is None
    assert ready[6:] == ([], [], "", "", "")
    with pytest.raises(gr.Error, match="Build"):
        web.compare_handler(None, "repo", "model.gguf", 4096, .7, 256, .95, "question")


def test_catalog_files_and_context(monkeypatch, session):
    monkeypatch.setattr(llm, "account_models", lambda: ["mine/model"])
    monkeypatch.setattr(llm, "search_models", lambda q: [q + "/found"])
    assert web.catalog_handler("My account")["value"] == "mine/model"
    assert web.catalog_handler("Search Hub", "tiny")["value"] == "tiny/found"
    assert "Llama-3.2-1B" in web.catalog_handler("Recommended", session=session)["value"]
    monkeypatch.setattr(llm, "gguf_files", lambda _: [
        {"filename": "Q4_K_M.gguf", "size": 1024**3},
        {"filename": "Q8.gguf", "size": 2 * 1024**3}])
    files = web.files_handler("repo")
    assert files["value"] == "Q4_K_M.gguf"
    assert "1.00 GiB" in files["choices"][0][0]
    assert web.files_handler(None)["value"] is None
    override, info = web.context_limit_handler("repo", "file", session, 16000)
    assert override["maximum"] == override["value"] == 8192
    assert "Pack context:" in info and "8,192" in info
    monkeypatch.setattr(llm, "account_models", list)
    assert web.catalog_handler("My account")["value"] is None
    with pytest.raises(gr.Error, match="Unknown"):
        web.catalog_handler("bad")
    monkeypatch.setattr(llm, "gguf_files", Mock(side_effect=RuntimeError("Hub unavailable")))
    with pytest.raises(gr.Error, match="Hub unavailable"):
        web.files_handler("repo")


def test_comparison_updates_both_histories_and_uses_safe_pack(monkeypatch, session):
    metadata = {"device": "cpu", **{side: {"prompt_tokens": 40, "generation_seconds": .5}
                                  for side in ("without_context", "with_context")}}
    worker = Mock(return_value={"without_context": "plain reply", "with_context": "cited reply",
                                "metadata": metadata})
    monkeypatch.setattr(gpu, "compare", worker)
    history = [{"role": "user", "content": [{"type": "text", "text": "Earlier question"}]},
               {"role": "assistant", "content": [{"type": "text", "text": "Earlier reply"}]}]
    result = web.compare_handler(session, "repo", "file", 4096, .7, 256, .95,
                                 "Question?", history, history)
    assert len(result[0]) == len(result[1]) == 4
    assert result[0][-1]["content"] == "plain reply"
    assert result[1][-1]["content"] == "cited reply"
    assert "cpu · 40 prompt tokens · 0.50s" in result[2]
    assert "Pending review" in result[5]
    args = worker.call_args.kwargs
    assert "ignore previous instructions" not in args["with_system"].lower()
    assert args["with_system"] != args["without_system"]
    assert args["with_history"][-1] == {"role": "user", "content": "Question?"}
    assert args["with_history"][0]["content"] == "Earlier question"
    assert len(history) == 2
    with pytest.raises(gr.Error, match="training context"):
        web.compare_handler(session, "repo", "file", 16000, .7, 256, .95, "Q?")
    with pytest.raises(gr.Error, match="question"):
        web.compare_handler(session, "repo", "file", 4096, .7, 256, .95, " ")
    with pytest.raises(gr.Error, match="Select"):
        web.compare_handler(session, "", "", 4096, .7, 256, .95, "Q?")
    worker.side_effect = RuntimeError("Cannot load llama.cpp")
    with pytest.raises(gr.Error, match="Cannot load llama.cpp"):
        web.compare_handler(session, "repo", "file", 4096, .7, 256, .95, "Q?")
    assert web.clear_compare() == ([], [], "", "", "")


def test_gpu_registration(monkeypatch):
    decorated, invoked = [], []

    def decorator(**options):
        decorated.append(options)

        def wrap(fn):
            def run(**kwargs):
                invoked.append(kwargs)
                return fn(**kwargs)
            return run
        return wrap

    monkeypatch.setitem(sys.modules, "spaces", SimpleNamespace(GPU=decorator))
    monkeypatch.setattr(llm, "compare_chat", lambda **kwargs: kwargs)
    spec = importlib.util.spec_from_file_location("agentmilieu._test_gpu", Path(gpu.__file__))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.compare(repo="repo") == {"repo": "repo"}
    assert decorated == [{"duration": 120}] and invoked == [{"repo": "repo"}]


def test_unraisable_hook_is_narrow(monkeypatch):
    monkeypatch.setattr(web, "create_app", lambda: None)
    delegate = Mock()
    monkeypatch.setattr(sys, "__unraisablehook__", delegate)
    monkeypatch.setattr(sys, "unraisablehook", sys.unraisablehook)
    spec = importlib.util.spec_from_file_location("_test_entrypoint", Path(__file__).parents[1] / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ignored = SimpleNamespace(object=BaseEventLoop.__del__,
                              exc_value=ValueError("Invalid file descriptor: -1"))
    module.unraisable_hook(ignored)
    delegate.assert_not_called()
    for obj, exc in [(lambda: None, ignored.exc_value),
                     (BaseEventLoop.__del__, ValueError("different")),
                     (BaseEventLoop.__del__, RuntimeError("Invalid file descriptor: -1"))]:
        event = SimpleNamespace(object=obj, exc_value=exc)
        module.unraisable_hook(event)
        delegate.assert_called_with(event)


def test_compare_wiring():
    demo = web.create_app()
    assert {fn.api_name for fn in demo.fns.values()} >= {"compare", "compare_submit"}
    assert sum(isinstance(block, gr.Chatbot) for block in demo.blocks.values()) == 2
