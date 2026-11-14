from types import SimpleNamespace

from typer.testing import CliRunner

from agentmilieu.cli import app
from agentmilieu.tokens import HeuristicTokenCounter, HFTokenCounter


def test_heuristic():
    counter = HeuristicTokenCounter()
    assert counter.count("") == 0
    assert counter.count("hello") == 2
    assert counter.count("abcd") == 1


def test_hf_pluggable():
    class Tokenizer:
        def encode(self, text, add_special_tokens):
            assert add_special_tokens is False
            return SimpleNamespace(ids=text.split())
    counter = HFTokenCounter(Tokenizer())
    assert counter.count("") == 0
    assert counter.count("hello world") == 2


def test_cli():
    runner = CliRunner()
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "Knowledge budget=104 tokens" in result.stdout
    result = runner.invoke(app, ["--model", "unknown"])
    assert result.exit_code != 0
