import json
import zipfile
from pathlib import Path
from unittest.mock import Mock

import gradio as gr
import pytest

from agentmilieu import interfaces as ui
from agentmilieu import web
from agentmilieu.budget import BudgetConfig, calculate_budget
from agentmilieu.composer import ComposeOptions, compose
from agentmilieu.devices import DEVICE_PRESETS
from agentmilieu.documents import Document
from agentmilieu.models import MODEL_PRESETS
from agentmilieu.safety import SafetyPolicy, safety_check
from agentmilieu.tokens import HFTokenCounter


@pytest.fixture(autouse=True)
def web_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MILIEU_HOME", str(tmp_path / "registry"))
    monkeypatch.setattr(ui, "load_dotenv", lambda: None)
    monkeypatch.setattr(ui, "select_model", Mock(return_value=MODEL_PRESETS["Llama-3.2-1B"]))
    monkeypatch.setattr(web, "gguf_token_counter", lambda _: None)
    monkeypatch.setattr(web.gpu, "probe", lambda: {"error": "No GPU available"})
    monkeypatch.setattr(web.devices, "probe_cpu", lambda: DEVICE_PRESETS["phone-6gb"])
    for key in ("CONTEXT_DEV_API_KEY", "MISTRAL_API_KEY", "NOTION_TOKEN", "NOTION_DATABASE_ID"):
        monkeypatch.delenv(key, raising=False)


def build(**kwargs):
    return web.build_handler(**{"hf_repo": "bartowski/Llama-3.2-1B-Instruct-GGUF",
                               "hf_file": "model-Q4_K_M.gguf", **kwargs})


def test_ui_offline_build_review_export(tmp_path, monkeypatch):
    from agentmilieu import safety

    monkeypatch.setenv("MISTRAL_API_KEY", "fake")
    monkeypatch.setenv("NOTION_TOKEN", "fake")
    monkeypatch.setenv("NOTION_DATABASE_ID", "fake")
    classifier = Mock(side_effect=AssertionError("No external classification in offline mode"))
    monkeypatch.setattr(safety, "mistral_classify", classifier)
    files = [str(path) for path in Path("examples").glob("*.md")]
    result = build(files=files)
    session = result[0]
    assert "Pending review" in result[8] and "local" in result[8]
    assert result[9]["interactive"] is True
    assert {row[0] for row in result[5]} >= {"EMAIL", "SK_KEY", "policy"}
    assert "ignore previous instructions" not in result[6].lower()
    assert "100.0/100" in result[7]
    classifier.assert_not_called()
    with pytest.raises(gr.Error, match="Approved"):
        web.export_handler(session)
    assert "Approved" in web.decision_handler(session, "Approved")[0]
    archive = web.export_handler(session)
    with zipfile.ZipFile(archive) as zipped:
        manifest = json.loads(zipped.read("manifest.json"))
        assert manifest["status"] == "Approved" and "run.sh" in zipped.namelist()
    assert "Rejected" in web.decision_handler(session, "Rejected")[0]
    with pytest.raises(gr.Error, match="Approved"):
        web.export_handler(session)
    other = build(files=files)[0]
    assert other["root"] != session["root"] and other["pack_id"] != session["pack_id"]
    assert "Pending review" in web.refresh_handler(other)


@pytest.mark.parametrize("kwargs,match", [({}, "at least one"),
    ({"urls": "https://example.org"}, "Offline"),
    ({"hf_file": ""}, "Select"), ({"scorton": True}, "Offline"),
    ({"device_mode": "unknown"}, "selection"),
    ({"quality_cap": -1}, "Caps"), ({"kv_type": "bad"}, "kv_type"),
])
def test_ui_build_errors(kwargs, match):
    with pytest.raises(gr.Error, match=match):
        build(**kwargs)


def test_ui_empty_review_and_notion_guard(tmp_path, monkeypatch):
    with pytest.raises(gr.Error, match="Build"):
        web.refresh_handler(None)
    with pytest.raises(gr.Error, match="Build"):
        web.export_handler(None)
    result = build(files=["examples/specs.md"])
    session = result[0]
    registry = web.session_registry(session)
    registry.update_entry(session["pack_id"], backend="notion", page_id="page")
    notion = Mock()
    notion.get_status.return_value = "Pending review"
    monkeypatch.setattr(ui, "NotionStore", lambda: notion)
    assert "page" in web.refresh_handler(session)
    with pytest.raises(gr.Error, match="Notion"):
        web.decision_handler(session, "Approved")
    with pytest.raises(gr.Error, match="Choose"):
        web.decision_handler(session, "invalid")


def test_manual_ui_and_ram_split():
    result = build(files=["examples/specs.md"], device_mode="Manual")
    assert sum(row[1] for row in result[2]) == pytest.approx(6, abs=.01)
    assert "KV for context" in [row[0] for row in result[2]]


def test_app_wiring_and_banner(monkeypatch):
    monkeypatch.setattr(ui.shutil, "which", lambda _: None)
    forbidden_probe = Mock(side_effect=AssertionError("No startup probing"))
    monkeypatch.setattr(web.gpu, "probe", forbidden_probe)
    assert "unavailable" in ui.integrations()["Scorton"]
    demo = web.create_app()
    assert {fn.api_name for fn in demo.fns.values()} >= {"build", "status", "approve", "reject", "export"}
    assert any(isinstance(block, gr.State) for block in demo.blocks.values())
    sources = [b for b in demo.blocks.values() if isinstance(b, gr.Radio) and b.label == "Source"]
    assert len(sources) == 2 and sources[0].choices == sources[1].choices
    assert not any(isinstance(b, gr.Radio) and b.label == "Model selection" for b in demo.blocks.values())
    device = next(b for b in demo.blocks.values() if isinstance(b, gr.Radio) and b.label == "Device selection")
    assert device.value == "Probe this machine"
    assert "probe" in {fn.api_name for fn in demo.fns.values()}
    forbidden_probe.assert_not_called()


def test_exact_target_selection_and_counter_are_used(monkeypatch):
    tokenizer = Mock()
    tokenizer.encode.side_effect = lambda text, **_: Mock(ids=text.split())
    counter = HFTokenCounter(tokenizer)
    monkeypatch.setattr(web, "gguf_token_counter", lambda _: counter)
    repo = "private/exact-GGUF"
    file_choices = [("Q6 · 2 GiB", "exact-Q6.gguf"), ("Q4 · 1 GiB", "exact-Q4.gguf")]
    result = build(files=["examples/specs.md"], hf_repo=repo, hf_file="exact-Q6.gguf",
                   model_source="My account", model_repo_choices=[repo, "private/other"],
                   model_file_choices=file_choices)
    ui.select_model.assert_called_once_with(hf_repo=repo, hf_file="exact-Q6.gguf")
    assert "Hub tokenizer.json" in result[1] and "counts are estimated" not in result[1]
    pack = web._session_pack(result[0])
    assert pack["token_accounting"]["knowledge"] == counter.count(pack["knowledge"])
    setup = web.compare_setup(result[0])
    assert setup[2]["value"] == pack["budget"]["context_window"]
    assert setup[3]["choices"] == [repo, "private/other"] and setup[3]["value"] == repo
    assert setup[4] == "My account"
    assert setup[5]["choices"] == file_choices and setup[5]["value"] == "exact-Q6.gguf"


def test_tokenizer_fallback_warning_and_probe_summary():
    result = build(files=["examples/specs.md"])
    assert "token counts are estimated; the with-context prompt may exceed the budget" in result[1]
    assert "Budget uses:" in result[1] and "CPU RAM:" in result[1]
    assert "Budget uses:" in web.probe_handler()


def test_probe_allocation_failure_uses_cpu(monkeypatch):
    monkeypatch.setattr(web.gpu, "probe", Mock(side_effect=RuntimeError("GPU quota exhausted")))
    assert "GPU quota exhausted" in web.probe_handler()
    assert "Budget uses:" in web.probe_handler()


@pytest.mark.parametrize("environment,expected", [
    ({}, True),
    ({"CONTEXT_DEV_API_KEY": "configured"}, False),
    ({"MISTRAL_API_KEY": "configured"}, False),
    ({"NOTION_TOKEN": "configured", "NOTION_DATABASE_ID": "configured"}, False),
    ({"NOTION_TOKEN": "configured"}, True),
    ({"NOTION_DATABASE_ID": "configured"}, True),
])
def test_offline_checkbox_default_tracks_integrations(monkeypatch, environment, expected):
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    demo = web.create_app()
    checkbox = next(block for block in demo.blocks.values()
                    if isinstance(block, gr.Checkbox) and block.label.startswith("Offline mode"))
    assert checkbox.value is expected


def test_strategy_overrides_and_mistral_disable(monkeypatch):
    from agentmilieu import safety

    monkeypatch.setenv("MISTRAL_API_KEY", "fake")
    forbidden = Mock(side_effect=AssertionError("Disabled"))
    monkeypatch.setattr(safety, "mistral_classify", forbidden)
    report = safety_check(Document.create("Safe battery specifications."), policy=SafetyPolicy(use_mistral=False))
    forbidden.assert_not_called()
    model, device = MODEL_PRESETS["Llama-3.2-1B"], DEVICE_PRESETS["phone-6gb"]
    budget = calculate_budget(model, device)
    for strategy in ("stuff", "compress", "rag"):
        pack = compose(report.chunks, budget, sources=[report.source],
                       options=ComposeOptions(strategy=strategy, use_mistral=False))
        assert pack.strategy == strategy
    tiny = calculate_budget(model, device, BudgetConfig(max_ctx=2048))
    huge = safety_check(Document.create("Battery specifications and safe operation. " * 2000),
                        policy=SafetyPolicy(use_mistral=False))
    with pytest.raises(ValueError, match="Stuff override"):
        compose(huge.chunks, tiny, sources=[huge.source], options=ComposeOptions(strategy="stuff"))
    with pytest.raises(ValueError, match="RAG override"):
        compose(report.chunks, tiny, sources=[report.source],
                options=ComposeOptions(strategy="rag", retrieval_tokens=10000))
