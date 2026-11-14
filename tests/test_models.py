import io

import gguf
import pytest

from agentmilieu.models import MODEL_PRESETS, ModelSpec, _read_header


@pytest.fixture
def tiny_gguf(tmp_path):
    path = tmp_path / "tiny.gguf"
    writer = gguf.GGUFWriter(str(path), "llama")
    writer.add_name("tiny")
    writer.add_block_count(2)
    writer.add_head_count(4)
    writer.add_head_count_kv(2)
    writer.add_embedding_length(32)
    writer.add_context_length(512)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def test_local_gguf(tiny_gguf):
    model = ModelSpec.from_gguf(tiny_gguf)
    assert (model.n_layers, model.n_kv_heads, model.head_dim, model.n_ctx_train) == (2, 2, 8, 512)
    assert model.weights_bytes == tiny_gguf.stat().st_size
    assert model.name == "tiny"


def test_remote_header_only(monkeypatch, tiny_gguf):
    import huggingface_hub

    raw = tiny_gguf.read_bytes()
    stream = io.BytesIO(raw + b"tensor payload must not be read")
    class FakeFS:
        def info(self, path):
            assert path == "owner/repo/model.gguf"
            return {"size": 10**9}

        def open(self, path, mode, **kwargs):
            assert kwargs == {"block_size": 64 << 10, "cache_type": "blockcache"}
            return stream

    monkeypatch.setattr(huggingface_hub, "HfFileSystem", FakeFS)
    remote = ModelSpec.from_hub("owner/repo", "model.gguf")
    assert remote.weights_bytes == 10**9
    assert remote.head_dim == 8


def test_header_stops_before_payload(tiny_gguf):
    stream = io.BytesIO(tiny_gguf.read_bytes() + b"payload")
    assert _read_header(stream, 4096)["llama.block_count"] == 2
    assert stream.tell() <= tiny_gguf.stat().st_size
    with pytest.raises(ValueError, match="limit"):
        _read_header(io.BytesIO(tiny_gguf.read_bytes()), 25)


@pytest.mark.parametrize("data", [b"", b"oops", b"GGUF\x01\x00\x00\x00"])
def test_invalid_header(data):
    with pytest.raises(ValueError):
        _read_header(io.BytesIO(data), 4096)


def test_metadata_fallback_and_explicit_key():
    metadata = {"general.architecture": "odd", "odd.block_count": 2,
                "odd.attention.head_count": 4, "odd.embedding_length": 32,
                "odd.context_length": 128}
    assert ModelSpec.from_metadata(metadata, 100).n_kv_heads == 4
    metadata["odd.attention.key_length"] = 16
    del metadata["odd.embedding_length"]
    assert ModelSpec.from_metadata(metadata, 100).head_dim == 16
    metadata["odd.attention.value_length"] = 8
    with pytest.raises(ValueError, match="Unequal"):
        ModelSpec.from_metadata(metadata, 100)


def test_missing_metadata_and_invalid_manual():
    with pytest.raises(ValueError, match="architecture"):
        ModelSpec.from_metadata({}, 100)
    with pytest.raises(ValueError, match="head_count"):
        ModelSpec.from_metadata({"general.architecture": "odd"}, 100)
    with pytest.raises(ValueError, match="n_layers"):
        ModelSpec(0, 1, 1, 1, 1)
    assert len(MODEL_PRESETS) == 8
    assert all(model.approximate for model in MODEL_PRESETS.values())
