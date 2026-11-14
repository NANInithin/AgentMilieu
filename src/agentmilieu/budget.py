"""Conservative estimates, not guarantees of fit or latency on a particular runtime."""

import math
from dataclasses import dataclass, field

from .devices import DeviceProfile
from .models import ModelSpec

KV_BYTES = {"f16": 2.0, "q8_0": 1.0625, "q4_0": 0.5625}


@dataclass(frozen=True)
class BudgetConfig:
    kv_type: str = "f16"
    usable_fraction: float = 0.7
    compute_buffer_bytes: int | None = None
    retrieval_overhead_bytes: int = 0
    max_ctx: int | None = None
    prefill_tok_s: float = 100.0
    target_first_token_s: float | None = None
    prompt_cache_path: str = "milieu-prompt.cache"
    quality_cap: int | None = 8192
    system_tokens: int = 256
    tool_tokens: int = 0
    query_tokens: int = 128
    retrieval_tokens: int = 0
    max_new_tokens: int = 512

    def __post_init__(self):
        if self.kv_type not in KV_BYTES:
            raise ValueError(f"kv_type must be one of {tuple(KV_BYTES)}")
        if not math.isfinite(self.usable_fraction) or not 0 < self.usable_fraction <= 1:
            raise ValueError("usable_fraction must be in (0, 1]")
        for name in ("prefill_tok_s", "target_first_token_s"):
            if name == "target_first_token_s" and self.target_first_token_s is None:
                continue
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("compute_buffer_bytes", "retrieval_overhead_bytes", "max_ctx", "quality_cap",
                     "system_tokens", "tool_tokens", "query_tokens", "retrieval_tokens", "max_new_tokens"):
            value = getattr(self, name)
            if value is None and name not in ("compute_buffer_bytes", "max_ctx", "quality_cap"):
                raise ValueError(f"{name} must be a non-negative integer")
            if value is not None and (not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or None")


@dataclass(frozen=True)
class BudgetResult:
    model: ModelSpec
    device: DeviceProfile
    config: BudgetConfig
    kv_bytes_per_element: float
    kv_bytes_per_token: float
    usable_ram_bytes: int
    compute_buffer_bytes: int
    free_bytes: int
    mem_tokens: int
    latency_cap: int | None
    estimated_cold_prefill_s: float
    constraints: dict[str, int]
    context_window: int
    reserved_tokens: int
    knowledge_budget: int
    binding_constraints: tuple[str, ...]
    llama_cpp_flags: tuple[str, ...]
    explanation: tuple[str, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)


def calculate_budget(model: ModelSpec, device: DeviceProfile,
                     config: BudgetConfig | None = None) -> BudgetResult:
    config = config or BudgetConfig()
    element = KV_BYTES[config.kv_type]
    kv = 2 * model.n_layers * model.n_kv_heads * model.head_dim * element
    usable = int(device.available_ram * config.usable_fraction)
    # Scratch/graph workspace heuristic: max(256 MiB, 10% of weights).
    compute = config.compute_buffer_bytes
    if compute is None:
        compute = max(256 * 1024**2, math.ceil(model.weights_bytes * 0.1))
    free = usable - model.weights_bytes - compute - config.retrieval_overhead_bytes
    memory = max(0, math.floor(free / kv))
    latency = (math.floor(config.prefill_tok_s * config.target_first_token_s)
               if config.target_first_token_s is not None else None)
    constraints = {"memory": memory, "training": model.n_ctx_train}
    if latency is not None:
        constraints["latency"] = latency
    if config.max_ctx is not None:
        constraints["user"] = config.max_ctx
    if config.quality_cap is not None:
        constraints["quality"] = config.quality_cap
    context = min(constraints.values())
    cold_prefill = context / config.prefill_tok_s
    binding = tuple(key for key, value in constraints.items() if value == context)
    reserved = (config.system_tokens + config.tool_tokens + config.query_tokens
                + config.max_new_tokens + config.retrieval_tokens)
    knowledge = max(0, context - reserved)
    warnings = []
    if free <= 0:
        warnings.append("Model does not fit: weights and overhead exhaust usable RAM.")
    if reserved > context:
        warnings.append("Reservations exceed context; reduce reservations or select a larger window.")
    if model.approximate:
        warnings.append("Model preset values are approximate; inspect the actual GGUF before export.")
    warnings.append("RAM, latency and quality are estimates; validate on the target device/runtime.")
    flags = (("-c", str(context), "-ctk", config.kv_type, "-ctv", config.kv_type,
              "-n", str(config.max_new_tokens), "--prompt-cache", config.prompt_cache_path,
              "--prompt-cache-ro") if context > 0 and reserved <= context else ())
    lines = (
        f"Model: {model.name}; device: {device.name}",
        (f"RAM total={device.total_ram:,}; available={device.available_ram:,} bytes; "
         f"usable={usable:,} ({config.usable_fraction:.0%} of available)."),
        (f"Weights={model.weights_bytes:,}; compute buffer={compute:,}; "
         f"retrieval/index={config.retrieval_overhead_bytes:,} bytes."),
        f"Free for KV={free:,} bytes; KV={kv:,.2f} bytes/token ({config.kv_type}).",
        "Context constraints: " + ", ".join(f"{k}={v:,}" for k, v in constraints.items()),
        f"Context window={context:,}; binding: {', '.join(binding)}.",
        f"Estimated cold prefill time={cold_prefill:.2f}s at {config.prefill_tok_s:g} tok/s.",
        (f"Reserve system={config.system_tokens}, tools={config.tool_tokens}, "
         f"query={config.query_tokens}, retrieval={config.retrieval_tokens}, "
         f"generation={config.max_new_tokens}; total={reserved:,}."),
        f"Knowledge budget={knowledge:,} tokens.",
        "Recommended llama.cpp flags: " + (" ".join(flags) if flags else "none (infeasible budget)"),
    )
    return BudgetResult(model, device, config, element, kv, usable, compute, free, memory,
                        latency, cold_prefill, constraints, context, reserved, knowledge, binding, flags,
                        lines, tuple(warnings))
