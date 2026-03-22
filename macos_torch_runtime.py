from __future__ import annotations

from contextlib import nullcontext

import torch


DEFAULT_MACOS_VAL_MAX_TOKENS = 16 * 1024


def choose_device_type(mps_available: bool, force_cpu: bool) -> str:
    if mps_available and not force_cpu:
        return "mps"
    return "cpu"


def get_device(force_cpu: bool = False) -> torch.device:
    device_type = choose_device_type(
        mps_available=torch.backends.mps.is_available(),
        force_cpu=force_cpu,
    )
    return torch.device(device_type)


def get_autocast_context(device_type: str, enabled: bool = True):
    if enabled and device_type == "cuda":
        return torch.autocast(device_type=device_type, enabled=enabled)
    return nullcontext()


def get_eval_context():
    return torch.no_grad()


def should_enable_compile(device_type: str, requested: bool) -> bool:
    return requested and device_type == "cpu"


def maybe_compile(model, device_type: str, requested: bool, log_fn):
    if not should_enable_compile(device_type, requested):
        return model
    try:
        return torch.compile(model)
    except Exception as exc:
        log_fn(f"torch.compile failed: {exc}")
        return model


def synchronize_device(device_type: str) -> None:
    if device_type == "mps":
        synchronize = getattr(torch.mps, "synchronize", None)
        if synchronize is not None:
            synchronize()


def peak_memory_summary(device_type: str) -> str:
    if device_type == "mps":
        return "peak memory allocated: not_supported on mps"
    return "peak memory allocated: unavailable on cpu"


def metric_accum_dtype(device_type: str) -> torch.dtype:
    if device_type == "mps":
        return torch.float32
    return torch.float64


def trim_validation_tokens_for_smoke(
    tokens: torch.Tensor,
    seq_len: int,
    val_max_tokens: int,
) -> torch.Tensor:
    if val_max_tokens <= 0:
        return tokens

    min_tokens = seq_len
    capped_tokens = max(val_max_tokens, min_tokens)
    usable_tokens = min(tokens.numel() - 1, (capped_tokens // seq_len) * seq_len)
    if usable_tokens <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable_tokens + 1]


def adam_optimizer_kwargs(device_type: str) -> dict[str, object]:
    if device_type == "cuda":
        return {"fused": True}
    return {}
