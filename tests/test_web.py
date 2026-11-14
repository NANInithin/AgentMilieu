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


@pytest.fixture(autouse=True)
def web_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MILIEU_HOME", str(tmp_path / "registry"))
    monkeypatch.setattr(ui, "load_dotenv", lambda: None)
    for key in ("CONTEXT_DEV_API_KEY", "MISTRAL_API_KEY", "NOTION_TOKEN", "NOTION_DATABASE_ID"):
        monkeypatch.delenv(key, raising=False)


def test_ui_offline_build_review_export(tmp_path, monkeypatch):
    from agentmilieu import safety

    monkeypatch.setenv("MISTRAL_API_KEY", "fake")
    monkeypatch.setenv("NOTION_TOKEN", "fake")
    monkeypatch.setenv("NOTION_DATABASE_ID", "fake")
    classifier = Mock(side_effect=AssertionError("No external classification in offline mode"))
    monkeypatch.setattr(safety, "mistral_classify", classifier)
    files = [str(path) for path in Path("examples").glob("*.md")]
    result = web.build_handler(files=files)
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
    other = web.build_handler(files=files)[0]
    assert other["root"] != session["root"] and other["pack_id"] != session["pack_id"]
    assert "Pending review" in web.refresh_handler(other)


@pytest.mark.parametrize("kwargs,match", [({}, "at least one"),
    ({"urls": "https://example.org"}, "Offline"),
    ({"model_mode": "HF GGUF"}, "Offline"), ({"scorton": True}, "Offline"),
    ({"model_mode": "unknown"}, "selection"), ({"device_mode": "unknown"}, "selection"),
    ({"quality_cap": -1}, "Caps"), ({"kv_type": "bad"}, "kv_type"),
])
def test_ui_build_errors(kwargs, match):
    with pytest.raises(gr.Error, match=match):
        web.build_handler(**kwargs)


def test_ui_empty_review_and_notion_guard(tmp_path, monkeypatch):
    with pytest.raises(gr.Error, match="Build"):
        web.refresh_handler(None)
    with pytest.raises(gr.Error, match="Build"):
        web.export_handler(None)
    result = web.build_handler(files=["examples/specs.md"])
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
    result = web.build_handler(files=["examples/specs.md"], model_mode="Manual", device_mode="Manual")
    assert sum(row[1] for row in result[2]) == pytest.approx(6, abs=.01)
    assert "KV for context" in [row[0] for row in result[2]]


def test_app_wiring_and_banner(monkeypatch):
    monkeypatch.setattr(ui.shutil, "which", lambda _: None)
    assert "unavailable" in ui.integrations()["Scorton"]
    demo = web.create_app()
    assert {fn.api_name for fn in demo.fns.values()} >= {"build", "status", "approve", "reject", "export"}
    assert any(isinstance(block, gr.State) for block in demo.blocks.values())


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
