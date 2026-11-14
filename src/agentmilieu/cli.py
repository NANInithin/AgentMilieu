"""Offline-first CLI for profiling and context budgeting."""

import typer

from .budget import BudgetConfig, calculate_budget
from .devices import DEVICE_PRESETS
from .models import MODEL_PRESETS

app = typer.Typer(help="AgentMilieu — a trusted context compiler for on-device SLMs")


@app.command()
def budget(model: str = "Llama-3.2-1B", device: str = "phone-6gb",
           max_ctx: int | None = None):
    """Explain a conservative context budget using approximate offline presets."""
    if model not in MODEL_PRESETS or device not in DEVICE_PRESETS:
        raise typer.BadParameter("Unknown model/device preset")
    try:
        result = calculate_budget(MODEL_PRESETS[model], DEVICE_PRESETS[device],
                                  BudgetConfig(max_ctx=max_ctx))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo("\n".join(result.explanation))
    for warning in result.warnings:
        typer.echo(f"Warning: {warning}")


if __name__ == "__main__":
    app()
