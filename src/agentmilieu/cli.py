"""Offline-first command line for context compilation and human review."""

from pathlib import Path
from typing import Annotated

import typer

from . import interfaces as ui
from .budget import BudgetConfig, calculate_budget
from .composer import ComposeOptions
from .devices import DEVICE_PRESETS
from .export import export as export_pack
from .models import MODEL_PRESETS
from .pipeline import BuildOptions, build_pack
from .safety import SafetyPolicy, redact_text

app = typer.Typer(help="AgentMilieu — a trusted context compiler for on-device SLMs")


def fail(exc):
    typer.echo(f"Error: {redact_text(str(exc))[0]}", err=True)
    return typer.Exit(1)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    ui.home()
    if ctx.invoked_subcommand is None:
        budget()


@app.command()
def budget(model: str = "Llama-3.2-1B", device: str = "phone-6gb",
           max_ctx: int | None = None):
    """Explain a conservative budget using approximate offline presets."""
    try:
        result = calculate_budget(ui.select_model(model), ui.select_device(device),
                                  BudgetConfig(max_ctx=max_ctx))
        typer.echo("\n".join(result.explanation))
        for warning in result.warnings:
            typer.echo(f"Warning: {warning}")
    except Exception as exc:
        raise fail(exc) from exc


@app.command()
def build(
    url: Annotated[list[str] | None, typer.Option()] = None,
    name: Annotated[list[str] | None, typer.Option(help="Research topic (Context.dev key required)")] = None,
    file: Annotated[list[Path] | None, typer.Option(help="File or directory; repeatable")] = None,
    preset: str | None = None, gguf: Path | None = None,
    hf_repo: str | None = None, hf_file: str | None = None,
    n_layers: int | None = None, n_kv_heads: int | None = None,
    head_dim: int | None = None, n_ctx_train: int | None = None, weights_gb: float | None = None,
    model_name: str = "manual", arch: str = "unknown",
    device_preset: str | None = None, ram_gb: float | None = None,
    available_gb: float | None = None, adb: bool = False,
    purpose: str = "", kv_type: str = "f16", max_ctx: int | None = None,
    quality_cap: int = 8192, latency_target: float | None = None, prefill_tok_s: float = 100,
    strategy: str = "auto", no_mistral: bool = False, offline: bool = False, scorton: bool = False,
):
    """Compile a Pending review draft. GB flags use GiB. Quality cap 0 disables the cap.

    --offline allows local files/models only and disables service calls.
    --no-mistral disables both the classifier and the summariser.
    """
    try:
        if offline and (url or name or hf_repo or hf_file or scorton):
            raise ValueError("--offline requires local files/model and no --scorton")
        model = ui.select_model(preset, gguf, hf_repo, hf_file, n_layers=n_layers,
                                n_kv_heads=n_kv_heads, head_dim=head_dim, n_ctx_train=n_ctx_train,
                                weights_gb=weights_gb, model_name=model_name, arch=arch)
        device = ui.select_device(device_preset, ram_gb, available_gb, adb)
        if quality_cap < 0:
            raise ValueError("quality-cap must be >= 0")
        disabled = no_mistral or offline
        options = BuildOptions(purpose=purpose, registry_root=str(ui.home()), use_notion=not offline,
            budget=BudgetConfig(kv_type=kv_type, max_ctx=max_ctx, quality_cap=quality_cap or None,
                                target_first_token_s=latency_target, prefill_tok_s=prefill_tok_s),
            composition=ComposeOptions(strategy=strategy, use_mistral=not disabled),
            safety=SafetyPolicy(use_mistral=not disabled), external_audit=scorton)
        result = build_pack(ui.source_inputs(url or [], name or [], file or []), model, device, options)
        typer.echo("\n".join(result.pack.budget.explanation))
        typer.echo(f"\nStrategy: {result.pack.strategy}\n" + "\n".join(result.pack.explanation))
        typer.echo("\nToken accounting\nSection             Tokens")
        for key, value in result.pack.token_accounting.items():
            typer.echo(f"{key:<20}{value:>8}")
        typer.echo("\nSafety findings (kind | severity | action | source | chunk | rule)")
        for row in ui.safety_rows(result.pack_id, ui.registry()):
            typer.echo(" | ".join(str(value) for value in row))
        typer.echo(f"Audit score: {result.audit['score']}/100")
        typer.echo(f"Scorton: {result.audit['external']['status']} — {result.audit['external']['reason']}")
        entry = ui.registry().get(result.pack_id)
        typer.echo(f"Pack ID: {result.pack_id}\nBackend: {result.backend}\nStatus: {result.status}")
        if entry["page_id"]:
            typer.echo(f"Notion page ID: {entry['page_id']}")
        for warning in (*result.pack.budget.warnings, *result.warnings):
            typer.echo(f"Warning: {warning}")
    except Exception as exc:
        raise fail(exc) from exc


@app.command()
def status(pack_id: str):
    """Read live Notion or local review status."""
    try:
        entry = ui.registry().get(pack_id)
        typer.echo(f"{pack_id}: {ui.status(pack_id, ui.registry())} ({entry['backend']})")
        if entry["page_id"]:
            typer.echo(f"Notion page ID: {entry['page_id']}")
    except Exception as exc:
        raise fail(exc) from exc


def decision(pack_id, value):
    try:
        ui.registry().set_status(pack_id, value)
        typer.echo(f"{pack_id}: {value}")
    except Exception as exc:
        raise fail(exc) from exc


@app.command()
def approve(pack_id: str):
    """Approve a local draft; review Notion-backed packs in Notion instead."""
    decision(pack_id, "Approved")


@app.command()
def reject(pack_id: str):
    """Reject a local draft; review Notion-backed packs in Notion instead."""
    decision(pack_id, "Rejected")


@app.command("export")
def export_command(pack_id: str, out: Path = Path("exports")):
    """Export only an Approved draft whose reviewed hash still matches."""
    try:
        typer.echo(f"Exported: {export_pack(pack_id, registry=ui.registry(), destination=out)}")
    except Exception as exc:
        raise fail(exc) from exc


@app.command()
def models():
    """List approximate Q4_K_M model presets."""
    for key, model in MODEL_PRESETS.items():
        typer.echo(f"{key}: {model.weights_bytes:,} bytes; {model.n_ctx_train:,} training tokens (approximate)")


@app.command()
def devices():
    """List illustrative RAM profiles (bytes)."""
    for key, device in DEVICE_PRESETS.items():
        typer.echo(f"{key}: total={device.total_ram:,}; available={device.available_ram:,}")


if __name__ == "__main__":
    app()
