# macOS Local Training Design

## Goal

Add a dedicated macOS local-development path that stays easy to rebase from `dev/4090-local-training`, keeps machine-specific paths private in `AGENTS.md`, and makes Apple Silicon smoke testing practical with both MLX and PyTorch.

## Constraints

- `dev/4090-local-training` remains the fast-moving CUDA-first branch.
- All committed public files must avoid machine-private paths, machine-private env values, and machine-private shell snippets.
- `train_gpt_mlx.py` is the primary macOS smoke path.
- PyTorch-on-macOS support is best-effort and optimized for local iteration, not leaderboard performance.
- macOS should avoid 4090-only assumptions such as CUDA-only autocast, NCCL setup, and 4090-specific `torch.compile` environment tricks.
- Success means tiny smoke runs complete on this Apple Silicon machine using MPS when available, otherwise CPU, while preserving parseable `val_bpb:` output.

## Approach

### Branching and integration

Create a dedicated `dev/macos-local-training` branch derived from the current 4090 branch. Keep macOS-specific work additive where possible so future rebases from `dev/4090-local-training` mostly involve replaying new files and a small number of targeted notes.

`dev/4090-local-training` remains the authoritative fast-iteration branch for CUDA work. The macOS branch must not force churn in CUDA-first files unless a tiny portable extraction clearly reduces maintenance cost.

### Runtime split

Use three entrypoints for local macOS work:

- `train_gpt_mlx.py` for the main Apple Silicon smoke workflow.
- `train_gpt_mac.py` for PyTorch-based local GPT smoke runs using MPS when available, CPU otherwise.
- `train_xt_mac.py` for PyTorch + x-transformers local smoke runs with the same MPS/CPU strategy.

The new PyTorch mac scripts should preserve familiar logging and environment variables where practical, but they should explicitly downgrade or disable CUDA-only features.

### Platform adaptation layer

Avoid editing the fast CUDA path in `train_gpt.py` and `train_xt.py` more than necessary. Instead, factor shared macOS-safe runtime helpers into a focused module that can answer questions like:

- Which device should be used (`mps` vs `cpu`)?
- Whether `torch.compile` should be attempted or skipped.
- Which autocast context should be used, if any.
- Whether distributed setup, fused optimizers, CUDA synchronization, and peak-memory logging need alternate behavior.

This keeps the new mac scripts readable and lowers rebase pain.

The shared extraction should stay limited to portable runtime selection and logging behavior. Existing CUDA scripts should not become dependent on macOS-specific branching unless needed for shared correctness. If a clean extraction is not practical for a given code path, limited duplication inside the new `*_mac.py` files is acceptable.

### Package and setup handling

Check the current macOS env using the repo-specific Python interpreter from `AGENTS.md`. Only codify portable package expectations in public files. If the repo needs a public package tweak, make it platform-conditional where possible.

### Private machine notes

Extend `AGENTS.md` with macOS-only notes such as:

- the local Python path,
- that MPS is available on this machine,
- that MLX is the preferred smoke path,
- that `torch.compile` behavior differs from CUDA and should be treated as optional,
- that 4090-specific env vars do not apply on macOS.

## Files

- Create: `train_gpt_mac.py`
- Create: `train_xt_mac.py`
- Create: `tests/test_macos_torch_runtime.py`
- Potentially create: a small shared helper module for macOS PyTorch runtime decisions
- Modify: `AGENTS.md`
- Possibly modify: `requirements.txt` if a portable, platform-conditional package fix is needed

## Verification

- Import-check `train_gpt_mlx.py`, `train_gpt_mac.py`, and `train_xt_mac.py`.
- Run platform-neutral unit tests for the macOS runtime helper behavior using mocked capability detection rather than requiring MPS hardware.
- Run a tiny `train_gpt_mlx.py` smoke command.
- Run tiny `train_gpt_mac.py` and `train_xt_mac.py` smoke commands on MPS or CPU.
- Confirm each smoke run emits a parseable `val_bpb:` line.
- Confirm the original CUDA-first scripts still import after any shared extraction.

## Non-goals

- No attempt to make macOS match 4090 throughput.
- No public documentation of private local paths.
- No broad rewrite of the existing CUDA scripts unless a tiny shared extraction clearly reduces maintenance cost.
