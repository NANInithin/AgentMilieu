"""Pending-review storage. Approval always binds an explicit, immutable draft hash."""

import hashlib
import json
import os
import time
import uuid
from pathlib import Path

STATUSES = ("Pending review", "Approved", "Rejected")
NOTION_SCHEMA = {
    "Name": {"title": {}},
    "Status": {"select": {"options": [{"name": name} for name in STATUSES]}},
    **{name: {"rich_text": {}} for name in ("Model", "Device", "Strategy", "Pack hash")},
    **{name: {"number": {}} for name in ("Context window", "Knowledge tokens", "Audit score")},
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(payload):
    return hashlib.sha256(canonical(payload).encode()).hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(canonical(value))
    temporary.replace(path)


class LocalRegistry:
    """Single-writer local registry. Use set_status to record a reviewer decision."""

    def __init__(self, root=".milieu"):
        self.root = Path(root).resolve()
        self.path = self.root / "registry.json"

    def _read(self):
        return json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    def get(self, pack_id):
        try:
            return self._read()[pack_id]
        except KeyError as exc:
            raise ValueError("Unknown pack_id") from exc

    def create(self, payload, metadata):
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        pack_id = uuid.uuid4().hex
        digest = content_hash(payload)
        _write_json(self.root / "drafts" / f"{pack_id}.json", payload)
        records = self._read()
        records[pack_id] = {"pack_hash": digest, "approved_hash": None, "status": "Pending review",
                            "backend": "local", "metadata": metadata, "page_id": None}
        _write_json(self.path, records)
        return pack_id

    def update_entry(self, pack_id, **values):
        records = self._read()
        if pack_id not in records:
            raise ValueError("Unknown pack_id")
        records[pack_id].update(values)
        _write_json(self.path, records)

    def get_status(self, pack_id):
        return self.get(pack_id)["status"]

    def set_status(self, pack_id, status):
        if status not in STATUSES:
            raise ValueError(f"Status must be one of {STATUSES}")
        record = self.get(pack_id)
        if record["backend"] != "local":
            raise ValueError("Review this pack in Notion; local overrides are not permitted")
        self.update_entry(pack_id, status=status,
                          approved_hash=record["pack_hash"] if status == "Approved" else None)

    def load_payload(self, pack_id):
        self.get(pack_id)
        # pack_id is selected from known entries, not accepted as a filesystem path.
        if not re_safe_id(pack_id):
            raise ValueError("Invalid pack_id")
        return json.loads((self.root / "drafts" / f"{pack_id}.json").read_text(encoding="utf-8"))


def re_safe_id(value):
    return len(value) == 32 and all(char in "0123456789abcdef" for char in value)


def notion_blocks(sections):
    blocks = []
    for title, text in sections:
        body = f"{title}\n{text}"
        for offset in range(0, len(body), 2000):
            blocks.append({"object": "block", "type": "paragraph", "paragraph": {
                "rich_text": [{"type": "text", "text": {"content": body[offset:offset + 2000]}}]}})
    return blocks


class NotionStore:
    """Uses Notion's pinned 2022-06-28 database API, including property updates.

    The raw update request is intentional: notion-client 3.x's database-update helper
    only accepts new API fields and drops legacy properties silently.
    """

    def __init__(self, database_id=None, *, client=None, sleep=time.sleep, retries=3):
        if client is None:
            from notion_client import Client
            client = Client(auth=os.environ["NOTION_TOKEN"], notion_version="2022-06-28")
        self.client = client
        self.database_id = database_id or os.environ["NOTION_DATABASE_ID"]
        self.sleep, self.retries = sleep, retries

    def _call(self, function, **kwargs):
        for attempt in range(self.retries + 1):
            try:
                return function(**kwargs)
            except Exception as exc:
                response = getattr(exc, "response", None)
                status = getattr(exc, "status", None) or getattr(response, "status_code", None)
                if status != 429 or attempt == self.retries:
                    raise
                headers = getattr(response, "headers", {}) or {}
                try:
                    delay = min(60, max(0, float(headers.get("Retry-After", 2**attempt))))
                except (ValueError, TypeError):
                    delay = 2**attempt
                self.sleep(delay)

    def ensure_schema(self):
        database = self._call(self.client.databases.retrieve, database_id=self.database_id)
        properties = database.get("properties", {})
        if "Name" not in properties and any("title" in value for value in properties.values()):
            raise ValueError("Rename the Notion database title property to Name")
        for name, schema in NOTION_SCHEMA.items():
            if name in properties and next(iter(schema)) not in properties[name]:
                raise ValueError(f"Notion property {name!r} has the wrong type")
        missing = {key: value for key, value in NOTION_SCHEMA.items() if key not in properties}
        if "Status" in properties:
            existing = properties["Status"]["select"].get("options", [])
            additions = [{"name": name} for name in STATUSES
                         if name not in {option["name"] for option in existing}]
            if additions:
                missing["Status"] = {"select": {"options": [*existing, *additions]}}
        if missing:
            self._call(self.client.request, path=f"databases/{self.database_id}", method="PATCH",
                       body={"properties": missing})

    def create(self, metadata, digest, sections):
        self.ensure_schema()
        values = dict(metadata, **{"Status": "Pending review", "Pack hash": digest})
        properties = {}
        for name, schema in NOTION_SCHEMA.items():
            kind = next(iter(schema))
            value = values[name]
            if kind in ("title", "rich_text"):
                properties[name] = {kind: [{"type": "text", "text": {"content": str(value)[:2000]}}]}
            elif kind == "select":
                properties[name] = {kind: {"name": value}}
            else:
                properties[name] = {kind: value}
        blocks = notion_blocks(sections)
        page = self._call(self.client.pages.create, parent={"database_id": self.database_id},
                          properties=properties, children=blocks[:100])
        for offset in range(100, len(blocks), 100):
            self._call(self.client.blocks.children.append, block_id=page["id"],
                       children=blocks[offset:offset + 100])
        return page["id"]

    def approval(self, page_id):
        properties = self._call(self.client.pages.retrieve, page_id=page_id)["properties"]
        status = (properties.get("Status", {}).get("select") or {}).get("name", "Pending review")
        digest = "".join(item.get("plain_text", item.get("text", {}).get("content", ""))
                         for item in properties.get("Pack hash", {}).get("rich_text", []))
        return status, digest

    def get_status(self, page_id):
        return self.approval(page_id)[0]
