"""Exercise the running local Gradio server with a session-scoped offline workflow."""

import json
import zipfile
from pathlib import Path

from gradio_client import Client, handle_file
from gradio_client.exceptions import AppError

from agentmilieu.interfaces import home


def main():
    client = Client("http://127.0.0.1:7860", download_files=home() / "verification-downloads",
                    analytics_enabled=False)
    examples = Path(__file__).resolve().parent
    client.predict(source="Recommended", query="", api_name="/target_catalog")
    repo = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
    selection = client.predict(repo=repo, api_name="/target_files")
    result = client.predict(urls="", names="", hf_repo=repo, hf_file=selection["value"],
        files=[handle_file(str(examples / name)) for name in ("specs.md", "workflow.md", "untrusted.md")],
        purpose="Explain offline use and safe review", offline=True, api_name="/build")
    print("Strategy:", result[2].splitlines()[0])
    print("Token accounting:", result[3]["data"])
    print("Safety findings:", len(result[4]["data"]))
    print(next(line for line in result[6].splitlines() if line.startswith("Score:")))
    assert "alice@example.com" not in result[5]
    assert "Ignore previous instructions" not in result[5]
    print(client.predict(api_name="/status"))
    try:
        client.predict(api_name="/export")
    except AppError as exc:
        assert "Approved" in str(exc)
        print("Before approval: REFUSED —", exc)
    else:
        raise AssertionError("Pending pack must not export")
    print(client.predict(api_name="/approve")[0])
    archive = client.predict(api_name="/export")
    with zipfile.ZipFile(archive) as zipped:
        manifest = json.loads(zipped.read("manifest.json"))
        assert manifest["status"] == "Approved"
        assert {"mcp_server.py", "run.sh", "audit.json", "context.md"} <= set(zipped.namelist())
    print("After approval: EXPORTED —", archive)
    print("Pack hash:", manifest["pack_hash"])


if __name__ == "__main__":
    main()
