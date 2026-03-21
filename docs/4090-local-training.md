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

## Deep Dive: Inductor Persistent Reduction Internals

This section documents the full investigation into why the OOM happens and what alternatives were tested, for anyone encountering similar issues on consumer GPUs.

### What Persistent Reduction Actually Does

Triton reduction kernels have two strategies:

- **Persistent**: loads the entire reduction dimension into shared memory, completes the reduction in one pass within a single kernel launch. Fast (no global memory round-trip), but SMEM usage scales with `reduction_dim × dtype_size × num_live_buffers`.
- **Non-persistent (multi-pass)**: processes the reduction in chunks, writing partial results to global memory between passes. Lower SMEM usage, but more memory traffic.

The inductor chooses persistent when `reduction_numel <= threshold` (default 1024 for inner reductions, 64 otherwise). This heuristic checks the **logical tile size** but not the **actual SMEM usage**, which also depends on how many tensors are live in the fused kernel.

### Why Manual RMSNorm Replacement Made It Worse

We tried replacing `F.rms_norm(x, ...)` with the equivalent manual ops:

```python
x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(x.dtype)
```

This **increased** the fused kernel size from 112 KB to 141 KB, because the decomposed ops gave inductor more individual nodes to fuse into a single persistent reduction kernel. The fused kernel name changed from `triton_per_fused__fused_rms_norm_...` (112 KB) to `triton_per_fused__to_copy__unsafe_view_add_div_expand_mul_pow_select_sum_unsqueeze_view_17` (141 KB).

Lesson: fighting the compiler's fusion decisions by decomposing ops often backfires.

### The mix_order_reduction Override Chain

The full chain that leads to the crash:

1. Inductor's scheduler identifies two reductions across different dimensions that share input tensors
2. `MixOrderReduction.can_fuse()` returns True, creating a `FusedMixOrderReductions` node
3. `_generate_kernel_code_for_mix_order_reduction()` creates the kernel with **hardcoded** `override_persistent_reduction=True`
4. The kernel constructor (`SIMDKernel.__init__`) sees the override and skips `should_use_persistent_reduction()` entirely
5. `_persistent_reduction_configs()` generates configs constrained by `XBLOCK * RBLOCK <= 4096` but **not** by actual SMEM
6. All generated configs exceed the 4090's 99 KB SMEM limit
7. `_make_launchers()` catches `OutOfResources` per-config but all fail → `RuntimeError: No valid triton configs`

### What Inductor Gets Wrong (Design Gap)

GEMM template kernels have `_prune_exceeding_max_shared_mem_configs()` which queries `shared_memory_per_block_optin` and filters configs before compilation. Reduction kernels have **no equivalent SMEM-aware pruning**. The `MAX_PERSISTENT_BLOCK_NUMEL = 4096` cap limits logical tile size but doesn't account for:

- dtype size of each live buffer
- Number of simultaneously live tensors in the fused kernel
- `num_stages` pipeline depth
- Register spill to SMEM

This is the root cause: the config generation doesn't know about hardware SMEM limits for reduction kernels.

### All Approaches Tested

| Approach | Result | Why |
|----------|--------|-----|
| `PERSISTENT_REDUCTIONS=0` | No effect | Bypassed by `mix_order_reduction`'s hardcoded override |
| `MULTI_KERNEL=1` | Crash | Triton `cache_key` bug in torch 2.10 (`_hash_lock` is None) |
| `MULTI_KERNEL=1` (even if it worked) | Would not help for MOR kernels | `optional_persistent` check skips fallback when `override_persistent_reduction=True` |
| Replace `F.rms_norm` with manual ops | Worse (141 KB > 112 KB) | More decomposed ops → larger fused kernel |
| `MAX_FUSION_SIZE=16` | No effect | Fusion size limit doesn't apply to MOR fusion path |
| `COMPILE_THREADS=0` (sync compile) | No effect | Ruled out subprocess env propagation as cause |
| **`MIX_ORDER_REDUCTION=0`** | **Works** | **Disables the entire MOR codepath; normal heuristics respect SMEM** |
| Eager mode (`TORCHDYNAMO_DISABLE=1`) | Works but 1.6× slower | No compilation at all |

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
