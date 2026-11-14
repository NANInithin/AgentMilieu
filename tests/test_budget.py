from dataclasses import replace

import pytest

from agentmilieu import BudgetConfig, DeviceProfile, ModelSpec, calculate_budget
from agentmilieu.devices import DEVICE_PRESETS
from agentmilieu.models import MODEL_PRESETS

MODEL = ModelSpec(2, 2, 8, 100000, 1000)
DEVICE = DeviceProfile(10**9, 10**9)
CONFIG = BudgetConfig(usable_fraction=1, compute_buffer_bytes=100, quality_cap=None,
                      prefill_tok_s=10**9, target_first_token_s=1,
                      system_tokens=10, tool_tokens=20, query_tokens=30, max_new_tokens=40)


def test_budget_math():
    result = calculate_budget(MODEL, DEVICE, replace(CONFIG, retrieval_overhead_bytes=200))
    assert result.kv_bytes_per_token == 128
    assert result.free_bytes == 10**9 - 1300
    assert result.mem_tokens == (10**9 - 1300) // 128
    assert result.context_window == 100000
    assert result.reserved_tokens == 100
    assert result.knowledge_budget == 99900
    assert result.binding_constraints == ("training",)
    assert result.llama_cpp_flags == ("-c", "100000", "-ctk", "f16", "-ctv", "f16", "-n", "40",
                                      "--prompt-cache", "milieu-prompt.cache", "--prompt-cache-ro")


@pytest.mark.parametrize("kind,element", [("f16", 2), ("q8_0", 1.0625), ("q4_0", .5625)])
def test_cache_types(kind, element):
    result = calculate_budget(MODEL, DEVICE, replace(CONFIG, kv_type=kind))
    assert result.kv_bytes_per_token == 64 * element


@pytest.mark.parametrize("config,device,expected", [
    (replace(CONFIG, max_ctx=1000), DEVICE, "user"),
    (replace(CONFIG, quality_cap=2000), DEVICE, "quality"),
    (replace(CONFIG, prefill_tok_s=50, target_first_token_s=2), DEVICE, "latency"),
    (CONFIG, DeviceProfile(10000, 10000), "memory"),
])
def test_binding(config, device, expected):
    assert calculate_budget(MODEL, device, config).binding_constraints == (expected,)


def test_tied_constraints():
    result = calculate_budget(MODEL, DEVICE, replace(CONFIG, max_ctx=100000))
    assert result.binding_constraints == ("training", "user")


def test_does_not_fit():
    result = calculate_budget(MODEL, DeviceProfile(1000, 1000), CONFIG)
    assert result.free_bytes == -100
    assert result.context_window == result.knowledge_budget == result.mem_tokens == 0
    assert "does not fit" in result.warnings[0]
    assert result.llama_cpp_flags == ()


def test_reservations_exceed_context():
    result = calculate_budget(MODEL, DEVICE, replace(CONFIG, max_ctx=10))
    assert result.knowledge_budget == 0
    assert result.llama_cpp_flags == ()
    assert any("Reservations exceed" in warning for warning in result.warnings)


def test_default_compute_and_preset():
    result = calculate_budget(MODEL_PRESETS["Llama-3.2-1B"], DEVICE_PRESETS["phone-6gb"])
    assert result.compute_buffer_bytes == 256 * 1024**2
    assert result.context_window == 8192
    assert result.knowledge_budget == 7296
    assert result.binding_constraints == ("quality",)
    assert result.latency_cap is None
    assert result.estimated_cold_prefill_s == 81.92


@pytest.mark.parametrize("kwargs", [{"kv_type": "bad"}, {"usable_fraction": 0},
                                   {"prefill_tok_s": float("nan")}, {"max_ctx": -1},
                                   {"system_tokens": None}, {"retrieval_overhead_bytes": None},
                                   {"max_new_tokens": -1}, {"compute_buffer_bytes": -1}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        BudgetConfig(**kwargs)
