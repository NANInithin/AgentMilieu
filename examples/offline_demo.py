"""Run from the repository root: python examples/offline_demo.py. No service calls."""

from pathlib import Path

from agentmilieu.composer import ComposeOptions
from agentmilieu.devices import DEVICE_PRESETS
from agentmilieu.export import ApprovalError, export
from agentmilieu.models import MODEL_PRESETS
from agentmilieu.pipeline import BuildOptions, build_pack
from agentmilieu.store import LocalRegistry


def main():
    root = Path(__file__).resolve().parent
    registry = LocalRegistry(root.parent / ".milieu" / "demo")
    options = BuildOptions(use_notion=False, composition=ComposeOptions(use_mistral=False))
    result = build_pack([root / "specs.md", root / "workflow.md", root / "untrusted.md"],
                        MODEL_PRESETS["Llama-3.2-1B"], DEVICE_PRESETS["phone-6gb"], options,
                        registry=registry, classifier=lambda _: 0)
    print(f"Strategy: {result.pack.strategy}")
    print(f"Token accounting: {result.pack.token_accounting}")
    print(f"Context window: {result.pack.budget.context_window}")
    print(f"Audit score: {result.audit['score']}/100")
    print(f"Redactions: {result.audit['redaction_counts']}")
    print(f"Quarantined chunks: {len(result.pack.quarantined)}")
    try:
        export(result.pack_id, registry=registry, destination=registry.root / "exports")
    except ApprovalError as exc:
        print(f"Before approval: REFUSED — {exc}")
    registry.set_status(result.pack_id, "Approved")
    path = export(result.pack_id, registry=registry, destination=registry.root / "exports")
    print(f"After approval: EXPORTED — {path}")
    print(f"Pack hash: {result.pack_hash}")


if __name__ == "__main__":
    main()
