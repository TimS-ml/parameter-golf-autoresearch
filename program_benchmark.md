# Benchmark: Running Competition Records on 1x RTX 4090

Run each `records/` submission as-is on our hardware to establish comparable
baseline scores. This gives us ground truth for how competition results
translate to a single-GPU setup.

## Hardware

- GPU: NVIDIA RTX 4090 eGPU (24 GB VRAM)
- See `AGENTS.md` for full machine-specific config

## Compute Equivalence

The competition runs on **8x H100 SXM for 10 min**. Measured throughput on
our hardware (from 20-min pilot runs):

| Setup | Measured tokens/sec | Time to reach 816M tokens |
|-------|-------------------|--------------------------|
| 8x H100 (competition) | ~1.35M | 10 min |
| 1x 4090 eGPU — baseline (9L, seq=1024) | ~598K | **~23 min** |
| 1x 4090 eGPU — heavy (7-stack, SmearGate) | ~317K | ~43 min |

Throughput varies significantly by submission (317K-598K tok/s). At **60 min**,
the fastest submissions see ~2.15B tokens (2.6x competition) and even the
slowest see ~1.14B (1.4x competition). This is enough for QAT convergence,
SWA averaging, and late-stage warmdown to take effect.

### Multi-checkpoint recording

We run a single **60-min** experiment with frequent validation and record
val_bpb at multiple wallclock checkpoints: **5 / 10 / 15 / 20 / 30 / 45 / 60 min**.
The full training curve is also exported to CSV for plotting.

The log lines include `train_time:<ms>`, so we post-process to extract the
val_bpb closest to each checkpoint.

| Checkpoint | train_time (ms) | Tokens (approx, baseline) | Notes |
|------------|----------------|--------------------------|-------|
| 5 min | 300,000 | ~180M | Quick screening |
| 10 min | 600,000 | ~360M | Early differentiation |
| 15 min | 900,000 | ~540M | |
| 20 min | 1,200,000 | ~718M | ~Competition token count |
| 30 min | 1,800,000 | ~1.1B | 1.3x competition |
| 45 min | 2,700,000 | ~1.6B | 2x competition |
| 60 min | 3,600,000 | ~2.2B | Full benchmark budget |

Set `VAL_LOSS_EVERY=50` for ~2.5 min granularity (~24 checkpoints in 60 min).

## Benchmark Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Wallclock | **60 min (3600s)** | Enough for QAT/SWA convergence; ~2x competition tokens |
| VAL_LOSS_EVERY | 50 | ~2.5 min granularity for multi-checkpoint extraction |
| GPU count | 1 | Single 4090 |
| Artifact limit | 16 MB | Same as competition |
| Dataset | FineWeb-10B SP1024 | Same as competition |
| Seed | 1337 | Fixed for reproducibility (default in all submissions) |
| Metric | val_bpb at 5/10/15/20/30/45/60 min + final quant | Learning curve per submission |

## Setup

### Pre-flight checklist

```bash
# 1. Check data
ls ./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin 2>/dev/null | wc -l
ls ./data/tokenizers/fineweb_1024_bpe.model

# 2. If no shards exist, download:
python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 20

# 3. Check deps
python -c "
import torch; print(f'torch {torch.__version__}, CUDA {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
import sentencepiece; print('sentencepiece OK')
import zstandard; print('zstandard OK')
from adam_atan2_pytorch import MuonAdamAtan2; print('adam_atan2_pytorch OK')
"

# 4. If missing deps:
# pip install zstandard  # needed by top submissions
```

## Running a Submission

Run each submission **foreground**, one at a time, so you can monitor the log,
detect crashes, and react. Do NOT use batch scripts or background processes
(see AGENTS.md).

For each submission in `records/track_10min_16mb/<submission>/`:

### Step 1: Create the run directory

```bash
RUN_NAME="<submission>"
RUN_DIR="./models/${RUN_NAME}"
mkdir -p "$RUN_DIR"
```

### Step 2: Copy the script

```bash
cp records/track_10min_16mb/<submission>/train_gpt.py ./bench_train.py
```

Note: `smeargate_orthoinit_muonwd` uses `train_gpt_v5.py` instead.

### Step 3: Run foreground

```bash
CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH="/home/tim/miniforge3/envs/torch/lib:$LD_LIBRARY_PATH" \
TORCHINDUCTOR_MIX_ORDER_REDUCTION=0 \
MAX_WALLCLOCK_SECONDS=3600 \
VAL_LOSS_EVERY=50 \
PYTHONUNBUFFERED=1 \
/home/tim/miniforge3/envs/torch/bin/python bench_train.py > "$RUN_DIR/bench.log" 2>&1
```

### Step 4: Save artifacts

After the run completes, save the model checkpoints alongside the log:

```bash
# Move model artifacts into the run dir
mv final_model.pt "$RUN_DIR/" 2>/dev/null
mv final_model.int8.ptz "$RUN_DIR/" 2>/dev/null

# Copy the training script for reproducibility
cp bench_train.py "$RUN_DIR/train_script.py"
```

### Step 5: Extract training curve to CSV

Parse the log into a CSV for plotting:

```bash
# Header
echo "step,train_time_ms,val_loss,val_bpb" > "$RUN_DIR/curve.csv"

# Extract all validation checkpoints
grep "^step:.*val_bpb:" "$RUN_DIR/bench.log" | \
  sed 's/step:\([0-9]*\)\/[0-9]* val_loss:\([0-9.]*\) val_bpb:\([0-9.]*\) train_time:\([0-9]*\)ms.*/\1,\4,\2,\3/' \
  >> "$RUN_DIR/curve.csv"
```

### Step 6: Extract summary results

```bash
# All val_bpb checkpoints with timestamps
grep "step:.*val_bpb:" "$RUN_DIR/bench.log"

# Peak memory
grep "peak memory allocated:" "$RUN_DIR/bench.log"

# Final post-quant metric
grep "final_int8_zlib_roundtrip " "$RUN_DIR/bench.log"
```

### Step 7: Record to results_benchmark.tsv

If grep output is empty, the run crashed. Read the traceback:
```bash
tail -50 "$RUN_DIR/bench.log"
```

For benchmark, only fix **platform adaptation bugs** (Triton SMEM, missing deps,
single-GPU compat). Do not fix model design bugs — just log as `crash`.

Record results in `results_benchmark.tsv`, then move to the next submission.

**Important notes:**
- Some submissions use `zstandard` — install with `pip install zstandard`
- Some submissions may not have `MAX_WALLCLOCK_SECONDS` — check how wallclock is controlled
- DDP code should gracefully handle single-GPU (WORLD_SIZE=1)
- If a submission OOMs on 24 GB, reduce batch size via env vars (e.g. `TRAIN_BATCH_TOKENS=262144`)
- Submissions with sliding-window eval at quant stage can take >45 min extra — may need to kill

## Output Directory Structure

Each run produces a self-contained directory under `./models/`:

```
models/
  NaiveBaseline/
    bench.log           # Full training log
    curve.csv           # step, train_time_ms, val_loss, val_bpb
    final_model.pt      # Full-precision model checkpoint
    final_model.int8.ptz # Quantized model artifact
    train_script.py     # Copy of the training script used
  LongContextSeq2048/
    ...
```

## Expected Differences from Competition

1. **More tokens at 60 min**: Baseline sees ~2.2B tokens vs ~816M in competition. Advanced submissions see ~1.1-1.6B. This gives techniques more room to converge.
2. **Different throughput profile**: 4090 has higher memory bandwidth per FLOP than H100 SXM. Bandwidth-bound ops (attention) may be relatively faster.
3. **No DDP overhead**: Single GPU avoids all-reduce communication cost, so each step is slightly more efficient per-GPU.
4. **Triton differences**: 4090 (SM89) has 99 KB SMEM vs H100's 228 KB. The `TORCHINDUCTOR_MIX_ORDER_REDUCTION=0` fix is required.
5. **Grad accumulation**: With 1 GPU, scripts that assumed 8-way DDP will use 8x grad accumulation. Same effective batch but more micro-steps per update.
6. **Warmdown schedule**: With `MAX_WALLCLOCK_SECONDS=3600`, warmdown occupies a smaller fraction of total training than the competition's 10-min setup. Submissions with `warmdown_iters` tuned for 10-min H100 runs may behave differently.

## Results: results_benchmark.tsv

Log results to `results_benchmark.tsv` (tab-separated). Do not commit this file.

```
submission	bpb_5m	bpb_10m	bpb_15m	bpb_20m	bpb_30m	bpb_45m	bpb_60m	bpb_final_quant	memory_gb	tokens_M	status	notes
```

Columns:
1. `submission` — directory name under `models/`
2. `bpb_5m` through `bpb_60m` — val_bpb at nearest checkpoint to each wallclock mark
3. `bpb_final_quant` — post int8+zlib (or int6+zstd) roundtrip val_bpb
4. `memory_gb` — peak VRAM in GB
5. `tokens_M` — total tokens processed (millions)
6. `status` — `ok`, `oom`, `crash`, `timeout`
7. `notes` — any issues encountered

Use `—` for missing checkpoints (e.g. if a run crashes before 30 min).

## Submissions to Benchmark

Priority order (by competition val_bpb, best first):

| # | Submission | Competition BPB | Key Techniques | Status |
|---|-----------|----------------|----------------|--------|
| 1 | `NaiveBaseline` | 1.2244 | Reference baseline | done |
| 2 | `LongContextSeq2048` | 1.2058 | seq=2048, lower LR | done |
| 3 | `TrainingOptSeq4096` | 1.2014 | seq=4096, Muon tuning | done |
| 4 | `SlidingWindowEval` | 1.1925 | Sliding window eval only | done |
| 5 | `SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit` | 1.1748 | 10L, overtone init | done |
| 6 | `MixedQuant_Int6Int8_SlidingWindow` | 1.1630 | MLP 3x, mixed quant | done |
| 7 | `Seq2048_FP16Emb_TunedLR` | 1.1598 | 10L, QAT int6, seq=2048 | done |
| 8 | `smeargate_orthoinit_muonwd` | 1.1556 | SmearGate, BigramHash | done |
| 9 | `MLP3x_QAT_Int6_SlidingWindow` | 1.1502 | 11L, int6 QAT, zstd-22 | done |
| 10 | `Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA` | 1.1458 | 7 stacked techniques | done |
| 11 | `10L_Int5MLP_MuonWD04_SWA50` | 1.1428 | int5/int6, 10L, SWA | done |
| 12 | `11L_EfficientPartialXSA_FA3_SWA120` | 1.1307 | 11L, efficient partial XSA, FA3, SWA/120 | **TODO** |
| 13 | `11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` | 1.1271 | 11L XSA4+EMA+int6 MLP3x, WD=0.04 | **TODO** |
| 14 | `11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` | 1.1248 | Partial RoPE, LN scale, EMA, XSA4, late QAT | **TODO** |
| 15 | `11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` | 1.1233 | Best: EMA, GPTQ-lite, warmdown3500, QAT@0.15 | **TODO** |

Skip these (less interesting or incomplete):
- `LowerLR` — pure LR sweep, subsumed by others
- `FP16Embed_WD3600` — subsumed by later submissions
- `10L_MixedPrecision` — subsumed by later submissions
- `WarmdownQuantization` — interesting idea but subsumed
- `LoRA_TTT` — experimental test-time training, not competitive
- `int6_STE QAT_ MLP_bigram _U_Net` — incomplete (no train_gpt.py)

## Lessons from 20-min Pilot Runs

Key findings from initial 20-min benchmark (see `results_benchmark.tsv` for raw data):

1. **No architecture differentiation at 20 min**: All submissions cluster at bpb 1.33-1.34 at the 20-min mark, essentially matching the baseline. Advanced techniques need more tokens to show benefit.

2. **Throughput varies 2x**: Baseline runs at ~598K tok/s (877ms/step) while the heaviest submission (Int6_MLP3x_SmearGate) runs at only ~317K tok/s (1657ms/step). Throughput cost must be justified by BPB improvement.

3. **Int6 QAT breaks at low token counts**: Submissions using int6 quantization-aware training show catastrophic quant roundtrip degradation (1.77-2.89 BPB) when trained for only 20 min. The QAT hasn't converged. This is the primary motivation for extending to 60 min.

4. **Sliding-window eval is extremely slow**: Submissions that use sliding-window evaluation at quant stage take >45 min extra beyond training. May need to skip or kill these.

5. **VRAM fits comfortably**: All submissions fit in 24 GB (range: 7.9-17.5 GB). No OOMs observed.

## Analysis

After running all benchmarks, compare:

1. **Learning curves**: Plot curve.csv for each submission. Which techniques improve fastest? Which need more tokens to shine?
2. **Relative ranking at each checkpoint**: Does the ranking at 10 min predict 60 min? At what point do rankings stabilize?
3. **Absolute BPB on 4090 vs competition**: With 60 min, baseline sees ~2.2B tokens. How close do we get to competition BPB?
4. **Throughput**: Which techniques hurt tokens/sec the most on 4090? Compare `tokens_M` across submissions.
5. **VRAM**: Which techniques fit comfortably in 24 GB vs needing adjustment?
6. **Scaling efficiency**: Do techniques that need many tokens (e.g. SWA, QAT) show steeper improvement curves from 20->60 min?
7. **Quantization gap**: Compare pre-quant vs post-quant BPB. At 60 min, does int6 QAT converge properly?
