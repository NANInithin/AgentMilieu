"""Hugging Face Gradio Space entrypoint; import does not launch a server."""

import sys
from asyncio import BaseEventLoop
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from agentmilieu.web import create_app


def unraisable_hook(event):
    if (event.object is BaseEventLoop.__del__ and type(event.exc_value) is ValueError
            and str(event.exc_value) == "Invalid file descriptor: -1"):
        return
    sys.__unraisablehook__(event)


sys.unraisablehook = unraisable_hook

demo = create_app()

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False, ssr_mode=False)
