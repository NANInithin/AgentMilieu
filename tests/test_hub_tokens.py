from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agentmilieu import tokens


@pytest.fixture(autouse=True)
def clear_cache():
    tokens._hub_counter.cache_clear()
    yield
    tokens._hub_counter.cache_clear()


def test_repo_tokenizer_preferred_and_cached(monkeypatch):
    counter = object()
    download = Mock(return_value="tokenizer.json")
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(tokens, "hf_hub_download", download)
    monkeypatch.setattr(tokens.HFTokenCounter, "from_file", lambda _: counter)
    monkeypatch.setattr(tokens, "HfApi", lambda **_: pytest.fail("GGUF repo tokenizer was sufficient"))
    assert tokens.gguf_token_counter("owner/gguf") is counter
    assert tokens.gguf_token_counter("owner/gguf") is counter
    download.assert_called_once_with(repo_id="owner/gguf", filename="tokenizer.json", token="test-token")


@pytest.mark.parametrize("base", ["owner/base", ["owner/missing", "owner/base"]])
def test_base_model_tokenizer(monkeypatch, base):
    seen = []
    counter = object()

    def download(repo_id, **_kwargs):
        seen.append(repo_id)
        if repo_id != "owner/base":
            raise OSError("not found")
        return "base-tokenizer.json"

    monkeypatch.setattr(tokens, "hf_hub_download", download)
    monkeypatch.setattr(tokens, "HfApi", lambda **_: SimpleNamespace(
        model_info=lambda _: SimpleNamespace(card_data={"base_model": base})))
    monkeypatch.setattr(tokens.HFTokenCounter, "from_file", lambda _: counter)
    assert tokens.gguf_token_counter("owner/gguf") is counter
    assert seen[0] == "owner/gguf" and seen[-1] == "owner/base"


@pytest.mark.parametrize("card", [None, {}, {"base_model": "gated/base"}, {"base_model": []}])
def test_missing_gated_or_invalid_tokenizer_falls_back(monkeypatch, card):
    monkeypatch.setattr(tokens, "hf_hub_download", lambda **_: "invalid-tokenizer.json")
    monkeypatch.setattr(tokens.HFTokenCounter, "from_file", Mock(side_effect=ValueError("invalid tokenizer")))
    monkeypatch.setattr(tokens, "HfApi", lambda **_: SimpleNamespace(
        model_info=lambda _: SimpleNamespace(card_data=card)))
    assert tokens.gguf_token_counter("owner/gguf") is None


def test_tokenizer_network_failure_falls_back(monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(tokens, "hf_hub_download", fail)
    monkeypatch.setattr(tokens, "HfApi", fail)
    assert tokens.gguf_token_counter("owner/gguf") is None


@pytest.mark.parametrize("failure", ["download", "tokenizer"])
def test_transient_failure_is_retried_and_only_success_is_cached(monkeypatch, failure):
    counter = object()
    download = Mock(return_value="tokenizer.json")
    load = Mock(return_value=counter)
    if failure == "download":
        download.side_effect = [OSError("Hub unavailable"), "tokenizer.json"]
    else:
        load.side_effect = [ValueError("invalid tokenizer"), counter]
    monkeypatch.setattr(tokens, "hf_hub_download", download)
    monkeypatch.setattr(tokens.HFTokenCounter, "from_file", load)
    monkeypatch.setattr(tokens, "HfApi", Mock(side_effect=OSError("metadata unavailable")))
    assert tokens.gguf_token_counter("owner/gguf") is None
    assert tokens._hub_counter.cache_info().currsize == 0
    assert tokens.gguf_token_counter("owner/gguf") is counter
    assert tokens.gguf_token_counter("owner/gguf") is counter
    assert download.call_count == 2
    assert tokens._hub_counter.cache_info().currsize == 1
