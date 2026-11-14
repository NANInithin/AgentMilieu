"""Hugging Face Gradio Space entrypoint; import does not launch a server."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from agentmilieu.web import create_app

demo = create_app()

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False, ssr_mode=False)
