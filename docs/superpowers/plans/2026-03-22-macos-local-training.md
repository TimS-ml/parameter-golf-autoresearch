# macOS Local Training Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a macOS-focused local development branch with stable MLX smoke runs, PyTorch-based `train_gpt_mac.py` and `train_xt_mac.py` entrypoints, and private `AGENTS.md` guidance for this machine.

**Architecture:** Keep `dev/4090-local-training` as the CUDA-first path and add macOS support as a thin, additive layer on `dev/macos-local-training`. Introduce a small shared PyTorch runtime helper for device, autocast, compile, synchronization, optimizer kwargs, and logging decisions, then build the new `*_mac.py` scripts on top of that helper while leaving the CUDA scripts largely untouched. If a clean shared extraction is not practical for a specific training-loop branch, allow limited duplication in the macOS scripts rather than forcing churn in the CUDA-first files.

**Tech Stack:** Python, PyTorch 2.10 with MPS, MLX, x-transformers, SentencePiece, unittest

---

## Chunk 1: Runtime helpers and tests

### Task 1: Add a shared macOS PyTorch runtime helper

**Files:**
- Create: `macos_torch_runtime.py`
- Create: `tests/__init__.py`
- Create: `tests/test_macos_torch_runtime.py`

- [ ] **Step 1: Write the failing test**

Create tests that cover these exact helper APIs:

```python
import unittest

from macos_torch_runtime import (
    choose_device_type,
    peak_memory_summary,
    should_enable_compile,
)


class MacOSTorchRuntimeTests(unittest.TestCase):
    def test_choose_device_prefers_mps_when_available(self):
        self.assertEqual(choose_device_type(mps_available=True, force_cpu=False), "mps")

    def test_choose_device_falls_back_to_cpu(self):
        self.assertEqual(choose_device_type(mps_available=False, force_cpu=False), "cpu")

    def test_compile_disabled_by_default_on_mps(self):
        self.assertFalse(should_enable_compile("mps", requested=True))

    def test_peak_memory_summary_marks_mps_as_not_supported(self):
        self.assertIn("not_supported", peak_memory_summary("mps"))
```

These tests must stay platform-neutral and runnable on any machine by passing explicit booleans or mocked capability values; they must not require live MPS hardware.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_macos_torch_runtime -v`
Expected: FAIL with `ModuleNotFoundError` for `macos_torch_runtime`

- [ ] **Step 3: Write minimal implementation**

Implement a focused helper with these exact functions:

- `choose_device_type(mps_available: bool, force_cpu: bool) -> str`
- `get_device(force_cpu: bool = False) -> torch.device`
- `get_autocast_context(device_type: str, enabled: bool = True)`
- `should_enable_compile(device_type: str, requested: bool) -> bool`
- `maybe_compile(model, device_type: str, requested: bool, log_fn)`
- `synchronize_device(device_type: str) -> None`
- `peak_memory_summary(device_type: str) -> str`
- `adam_optimizer_kwargs(device_type: str) -> dict[str, object]`

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_macos_torch_runtime -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add macos_torch_runtime.py tests/__init__.py tests/test_macos_torch_runtime.py
git commit -m "feat: add macOS torch runtime helpers"
```

## Chunk 2: New macOS GPT entrypoint

### Task 2: Add `train_gpt_mac.py`

**Files:**
- Create: `train_gpt_mac.py`
- Modify: `tests/test_macos_torch_runtime.py`

- [ ] **Step 1: Write the failing test**

Extend tests to verify GPT-entrypoint-specific helper behavior:

```python
def test_cpu_can_opt_in_to_compile(self):
    self.assertTrue(should_enable_compile("cpu", requested=True))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_macos_torch_runtime -v`
Expected: FAIL because the new helper behavior is missing

- [ ] **Step 3: Write minimal implementation**

Create `train_gpt_mac.py` as a macOS-oriented adaptation of `train_gpt.py` that:

- imports reusable model/data/quant helpers from `train_gpt.py` instead of forking the full file,
- uses the shared runtime helper,
- runs single-process local execution only,
- avoids CUDA-only imports and NCCL setup,
- uses MPS or CPU,
- makes `torch.compile` opt-in or best-effort,
- keeps local smoke defaults modest,
- preserves parseable `val_bpb:` output.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_macos_torch_runtime -v`
Expected: PASS

- [ ] **Step 5: Run a smoke import / tiny execution**

Run:

```bash
python -c "import train_gpt_mac; print('ok')"
RUN_ID=gpt_mac_smoke ITERATIONS=1 VAL_LOSS_EVERY=1 TRAIN_BATCH_TOKENS=4096 VAL_BATCH_SIZE=4096 MAX_WALLCLOCK_SECONDS=30 python train_gpt_mac.py
```

Expected: import succeeds; tiny run reaches a parseable `val_bpb:` log line on MPS if available, otherwise CPU.

- [ ] **Step 6: Commit**

```bash
git add train_gpt_mac.py tests/test_macos_torch_runtime.py
git commit -m "feat: add macOS gpt training entrypoint"
```

## Chunk 3: New macOS x-transformers entrypoint

### Task 3: Add `train_xt_mac.py`

**Files:**
- Create: `train_xt_mac.py`
- Modify: `tests/test_macos_torch_runtime.py`

- [ ] **Step 1: Write the failing test**

Extend tests for any remaining helper behavior required by the XT script:

```python
def test_cpu_peak_memory_summary_has_known_prefix(self):
    self.assertTrue(peak_memory_summary("cpu").startswith("peak memory allocated:"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_macos_torch_runtime -v`
Expected: FAIL because the XT-specific helper behavior is not implemented yet

- [ ] **Step 3: Write minimal implementation**

Create `train_xt_mac.py` as a macOS-oriented adaptation of `train_xt.py` that:

- imports reusable model/data/quant helpers from `train_xt.py` instead of forking the full file,
- keeps local `x-transformers` import behavior,
- uses the runtime helper for MPS/CPU,
- falls back cleanly when `adam-atan2-pytorch` is absent,
- keeps `val_bpb:` logging,
- avoids CUDA-only synchronization and memory reporting.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_macos_torch_runtime -v`
Expected: PASS

- [ ] **Step 5: Run smoke verification**

Run:

```bash
python -c "import train_xt_mac; print('ok')"
RUN_ID=xt_mac_smoke ITERATIONS=1 VAL_LOSS_EVERY=1 TRAIN_BATCH_TOKENS=4096 VAL_BATCH_SIZE=4096 MAX_WALLCLOCK_SECONDS=30 python train_xt_mac.py
```

Expected: import succeeds; tiny XT run reaches a parseable `val_bpb:` log line on MPS if available, otherwise CPU.

- [ ] **Step 6: Commit**

```bash
git add train_xt_mac.py tests/test_macos_torch_runtime.py
git commit -m "feat: add macOS x-transformers entrypoint"
```

## Chunk 4: MLX verification and private machine notes

### Task 4: Verify MLX smoke path and update macOS private notes

**Files:**
- Modify: `AGENTS.md`
- Modify: `requirements.txt` (only if a safe public package tweak is needed)

- [ ] **Step 1: Write the failing test**

No production-code test is required for `AGENTS.md`, but before editing any committed runtime file in this task, identify whether a behavior test is needed. If no runtime file changes are needed beyond docs/config, keep this task doc-only.

- [ ] **Step 2: Verify current MLX import baseline**

Run: `python -c "import train_gpt_mlx; print('ok')"`
Expected: PASS

- [ ] **Step 3: Run a tiny MLX smoke command**

Run:

```bash
RUN_ID=mlx_smoke ITERATIONS=1 TRAIN_BATCH_TOKENS=4096 VAL_LOSS_EVERY=1 VAL_BATCH_SIZE=4096 MAX_WALLCLOCK_SECONDS=30 python train_gpt_mlx.py
```

Expected: reaches a parseable `val_bpb:` line.

- [ ] **Step 4: Update private macOS notes**

Add an `AGENTS.md` section for this machine covering:

- the correct macOS Python path,
- that MLX is the preferred local smoke path,
- that MPS is available,
- that `torch.compile` is optional/best-effort on macOS,
- that 4090-specific CUDA env vars do not apply.

Only touch `requirements.txt` if a platform-conditional package fix is clearly useful and safe.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt
git commit -m "docs: add macOS local training notes"
```

Do not commit `AGENTS.md`; it is machine-specific and intentionally private.
