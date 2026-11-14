import re
from pathlib import Path
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from agentmilieu import interfaces as ui
from agentmilieu.cli import app
from agentmilieu.models import MODEL_PRESETS


@pytest.fixture(autouse=True)
def interface_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MILIEU_HOME", str(tmp_path / "registry"))
    for key in ("CONTEXT_DEV_API_KEY", "MISTRAL_API_KEY", "NOTION_TOKEN", "NOTION_DATABASE_ID"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(ui, "load_dotenv", lambda: None)


def test_cli_offline_review_export(tmp_path, monkeypatch):
    from agentmilieu import safety

    monkeypatch.setenv("MISTRAL_API_KEY", "fake")
    forbidden = Mock(side_effect=AssertionError("Must not call Mistral"))
    monkeypatch.setattr(safety, "mistral_classify", forbidden)
    source = tmp_path / "notes.md"
    source.write_text("# Device\nWorks offline. Contact alice@example.com.")
    runner = CliRunner()
    built = runner.invoke(app, ["build", "--file", str(source), "--offline"])
    assert built.exit_code == 0, built.output
    assert "Strategy: stuff" in built.output and "redact" in built.output
    assert "Audit score:" in built.output and "Token accounting" in built.output
    assert "alice@example.com" not in built.output
    forbidden.assert_not_called()
    pack_id = re.search(r"Pack ID: ([a-f0-9]{32})", built.output)[1]
    assert "Pending review" in runner.invoke(app, ["status", pack_id]).output
    out = tmp_path / "exports"
    assert runner.invoke(app, ["export", pack_id, "--out", str(out)]).exit_code == 1
    assert not out.exists()
    assert runner.invoke(app, ["approve", pack_id]).exit_code == 0
    assert runner.invoke(app, ["export", pack_id, "--out", str(out)]).exit_code == 0
    assert (out / pack_id / "manifest.json").is_file()
    assert runner.invoke(app, ["reject", pack_id]).exit_code == 0
    assert runner.invoke(app, ["export", pack_id, "--out", str(tmp_path / "other")]).exit_code == 1


@pytest.mark.parametrize("args", [
    ["build"], ["build", "--hf-repo", "owner/repo"],
    ["build", "--preset", "bad"], ["build", "--ram-gb", "6"],
    ["build", "--preset", "Llama-3.2-1B", "--gguf", "missing.gguf"],
    ["build", "--offline", "--url", "https://example.org"],
    ["build", "--offline", "--scorton"], ["build", "--strategy", "magic"],
    ["status", "missing"], ["approve", "missing"], ["reject", "missing"],
    ["export", "missing"], ["budget", "--model", "bad"],
])
def test_cli_errors(args):
    result = CliRunner().invoke(app, args)
    assert result.exit_code != 0
    assert "Error:" in result.output


def test_presets_and_target_selectors(monkeypatch):
    runner = CliRunner()
    assert "Llama-3.2-1B" in runner.invoke(app, ["models"]).output
    assert "phone-6gb" in runner.invoke(app, ["devices"]).output
    assert "Knowledge budget" in runner.invoke(app, ["budget"]).output
    model = MODEL_PRESETS["Llama-3.2-1B"]
    gguf = Mock(return_value=model)
    hub = Mock(return_value=model)
    monkeypatch.setattr(ui.ModelSpec, "from_gguf", gguf)
    monkeypatch.setattr(ui.ModelSpec, "from_hub", hub)
    assert ui.select_model(gguf="model.gguf") == model
    assert ui.select_model(hf_repo="owner/repo", hf_file="model.gguf") == model
    gguf.assert_called_once_with("model.gguf")
    hub.assert_called_once_with("owner/repo", "model.gguf")
    assert ui.select_model(n_layers=16, n_kv_heads=8, head_dim=64, n_ctx_train=4096,
                           weights_gb=1).weights_bytes == 1024**3
    with pytest.raises(ValueError, match="Manual model"):
        ui.select_model(n_layers=16)
    with pytest.raises(ValueError, match="one device"):
        ui.select_device("phone-6gb", ram_gb=6)
    monkeypatch.setattr(ui, "probe_adb", lambda: None)
    with pytest.raises(ValueError, match="adb unavailable"):
        ui.select_device(adb=True)
    monkeypatch.setattr(ui, "probe_adb", lambda: ui.select_device())
    assert ui.select_device(adb=True).total_ram == 6 * 1024**3
    assert ui.select_device(ram_gb=6, available_gb=4).available_ram == 4 * 1024**3


def test_directory_and_notion_review(tmp_path, monkeypatch):
    (tmp_path / "notes.md").write_text("Safe notes")
    (tmp_path / "ignored.py").write_text("not a document")
    (tmp_path / ".hidden.md").write_text("hidden")
    assert len(ui.source_inputs(files=[tmp_path, tmp_path / "notes.md"])) == 1
    with pytest.raises(ValueError, match="does not exist"):
        ui.source_inputs(files=[tmp_path / "absent.md"])
    registry = ui.registry()
    pack_id = registry.create({"files": {}}, {})
    registry.update_entry(pack_id, backend="notion", page_id="notion-page")
    notion = Mock()
    notion.get_status.return_value = "Approved"
    monkeypatch.setattr(ui, "NotionStore", lambda: notion)
    assert "Approved" in CliRunner().invoke(app, ["status", pack_id]).output
    result = CliRunner().invoke(app, ["approve", pack_id])
    assert result.exit_code == 1 and "Notion" in result.output
    assert registry.get(pack_id)["approved_hash"] is None


def test_directory_and_mixed_absolute_relative_files_deduplicate(tmp_path, monkeypatch):
    directory = tmp_path / "docs"
    directory.mkdir()
    source = directory / "notes.md"
    source.write_text("Safe battery specifications", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    sources = ui.source_inputs(files=[Path("docs"), source.resolve(), Path("docs/../docs/notes.md")])
    assert len(sources) == 1
    assert sources[0].value == str(source.resolve())
    result = CliRunner().invoke(app, ["build", "--file", "docs", "--file", str(source.resolve()), "--offline"])
    assert result.exit_code == 0, result.output
    assert "Pack ID:" in result.output
