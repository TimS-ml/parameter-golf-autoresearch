# Running Parameter Golf on RTX 4090 (Laptop, 16GB VRAM)

The baseline `train_gpt.py` targets 8×H100 SXM. This document covers what breaks on a single RTX 4090 and how to fix it.

## Quick Start

```bash
# Environment: any conda env with torch 2.10+, sentencepiece, numpy, tqdm, huggingface-hub, datasets

# Download dataset (10 shards ≈ 1B tokens; use --train-shards 80 for full 8B)
python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

# Train (uncapped wallclock, since 10-min cap is for 8×H100)
TORCHINDUCTOR_MIX_ORDER_REDUCTION=0 \
RUN_ID=baseline_4090 \
MAX_WALLCLOCK_SECONDS=0 \
VAL_LOSS_EVERY=2000 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Expected: ~3100ms/step, ~17h for 20k steps. Baseline val_bpb ≈ 1.22.

## The Problem: Triton Persistent-Reduction OOM

### Symptom

```
torch._inductor.exc.InductorError: RuntimeError: No valid triton configs.
OutOfMemoryError: out of resource: triton_per_fused__fused_rms_norm_...
Required: 112800  Hardware limit: 101376
```

### Root Cause

`torch.compile` via the inductor backend generates **persistent reduction** Triton kernels. These kernels load the entire reduction dimension into shared memory (SMEM) for a single-pass reduction — fast, but memory-hungry.

| GPU | SMEM per SM | Status |
|-----|-------------|--------|
| H100 SXM | 228 KB | Works |
| A100 | 164 KB | Works |
| **RTX 4090 (Ada Lovelace, sm_89)** | **99 KB** | **OOM at 112 KB** |

The fused backward kernel for RMSNorm + surrounding ops requires ~112 KB of shared memory, exceeding the 4090's 99 KB limit.

### Why `TORCHINDUCTOR_PERSISTENT_REDUCTIONS=0` Doesn't Work

PyTorch 2.10's inductor has a feature called **mix_order_reduction** (enabled by default). When it fuses reductions across different dimensions, it **hardcodes** `override_persistent_reduction=True` in the codegen path:

```python
# torch/_inductor/codegen/simd.py, line ~1585
kernel_kwargs = {
    "mix_order_reduction": True,
    "override_persistent_reduction": True,  # <-- bypasses the config flag
}
```

This override completely bypasses `config.triton.persistent_reductions`. There is also an `assert kernel.persistent_reduction` immediately after, confirming this is intentional and unconditional.

### Why `TORCHINDUCTOR_MULTI_KERNEL=1` Alone Doesn't Help Either

The `multi_kernel` feature generates both persistent and non-persistent variants, benchmarking at runtime. However, it explicitly skips generating a non-persistent fallback when `override_persistent_reduction` is set:

```python
# torch/_inductor/codegen/triton.py, line ~6095
optional_persistent = kernel.persistent_reduction and not kernel_kwargs.get(
    "override_persistent_reduction"  # True for mix_order_reduction → no fallback
)
```

### The Fix

```bash
TORCHINDUCTOR_MIX_ORDER_REDUCTION=0
```

This disables the mix_order_reduction feature entirely, so:
1. The hardcoded `override_persistent_reduction=True` path is never taken
2. Inductor falls back to the normal heuristic (`should_use_persistent_reduction()`), which respects SMEM constraints
3. Reductions that would exceed SMEM are automatically compiled as non-persistent (multi-pass) kernels

**Performance impact**: mix_order_reduction fuses cross-dimension reductions into a single kernel launch. Without it, these become separate kernels with extra global memory traffic. For dim=512, the overhead is minimal (~5%).

### Recommended Configuration for 4090

```bash
TORCHINDUCTOR_MIX_ORDER_REDUCTION=0    # Required: fixes the OOM
```

Note: `TORCHINDUCTOR_MULTI_KERNEL=1` would theoretically add non-persistent fallbacks for remaining persistent reductions, but it crashes on torch 2.10 due to a Triton `cache_key` bug (`'NoneType' object does not support the context manager protocol`). Do not use it.

## Other 4090-Specific Notes

### NCCL Symbol Errors

If you see `undefined symbol: ncclAlltoAll` when importing torch, this is caused by conflicting `nvidia-nccl-cu12` and `nvidia-nccl-cu13` pip packages overwriting each other. Fix:

```bash
pip uninstall -y nvidia-nccl-cu12 nvidia-nccl-cu13
pip install nvidia-nccl-cu13 --force-reinstall
```

### Single-GPU Gradient Accumulation

The script assumes `WORLD_SIZE` divides 8, using `grad_accum_steps = 8 // world_size`. With 1 GPU, this means 8 micro-batches of 64 sequences × 1024 tokens = 65,536 tokens each, fitting within 16 GB VRAM.

### Speed Comparison

Benchmarked on RTX 4090 Laptop (16GB), torch 2.10.0+cu130, 2000 iterations, seed 1337:

| Configuration | Step Time | 2k Steps | val_bpb @2k | Notes |
|---------------|-----------|----------|-------------|-------|
| **compiled, MOR=0** | **3112ms** | **1h 44m** | **1.2962** | Recommended |
| eager (no compile) | 4975ms | 2h 46m | — | 1.6× slower |
| compiled, MOR=0 + MK=1 | — | — | — | Crashes (triton bug) |

For full 20k iterations: compiled ~17h, eager ~28h.

`torch.compile` with `MIX_ORDER_REDUCTION=0` gives a **1.60× speedup** over eager mode. The compilation overhead is amortized within the first ~50 steps via warmup.

### Default Environment Variables for 4090

```bash
# Required
TORCHINDUCTOR_MIX_ORDER_REDUCTION=0

# Recommended defaults for local iteration
ITERATIONS=2000          # ~1h44m instead of ~17h
MAX_WALLCLOCK_SECONDS=0  # disable 10-min cap (meant for 8×H100)
VAL_LOSS_EVERY=2000      # validate only at end
TRAIN_LOG_EVERY=200      # reduce log noise
```
