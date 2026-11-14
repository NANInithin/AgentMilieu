"""The only public exporter: review status and approved content must both match."""

import hashlib
import hmac
import json
import shutil
import tempfile
from pathlib import Path

from .store import LocalRegistry, NotionStore, content_hash


class ApprovalError(ValueError):
    pass


def export(pack_id, *, registry=None, destination="exports", notion=None):
    registry = registry or LocalRegistry()
    entry = registry.get(pack_id)
    status, approved_hash = entry["status"], entry.get("approved_hash")
    if entry["backend"] == "notion":
        try:
            status, approved_hash = (notion or NotionStore()).approval(entry["page_id"])
        except Exception as exc:
            raise ApprovalError("Cannot verify Notion approval; export refused") from exc
    if status != "Approved":
        raise ApprovalError(f"Export requires Approved status; current status: {status}")
    payload = registry.load_payload(pack_id)
    actual = content_hash(payload)
    if (not approved_hash or not hmac.compare_digest(actual, approved_hash)
            or not hmac.compare_digest(actual, entry["pack_hash"])):
        raise ApprovalError("Pack hash does not match reviewed content (tamper check failed)")
    files = payload["files"]
    if any(Path(name).name != name or name in (".", "..", "manifest.json") for name in files):
        raise ApprovalError("Invalid artifact path in reviewed pack")
    destination = Path(destination).resolve()
    target = destination / pack_id
    if target.exists():
        raise FileExistsError(f"Export already exists: {target}")
    destination.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".export-", dir=destination))
    try:
        encoded = {name: text.encode("utf-8") for name, text in files.items()}
        for name, data in encoded.items():
            (stage / name).write_bytes(data)
        manifest = {"pack_id": pack_id, "pack_hash": actual, "status": "Approved",
                    "metadata": payload["metadata"], "pack": payload["pack"],
                    "hashes": {name: hashlib.sha256(data).hexdigest() for name, data in encoded.items()},
                    "hash_scope": "Pack hash covers canonical draft payload; file hashes exclude manifest itself."}
        (stage / "manifest.json").write_bytes(json.dumps(manifest, indent=2).encode("utf-8"))
        (stage / "run.sh").chmod(0o700)
        stage.rename(target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return target
