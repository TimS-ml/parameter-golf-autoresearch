# Benchmark: Running Competition Records on 1x RTX 4090

Run each `records/` submission as-is on our hardware to establish comparable
baseline scores. This gives us ground truth for how competition results
translate to a single-GPU setup.

## Hardware

- GPU: NVIDIA RTX 4090 eGPU (24 GB VRAM)
- See `AGENTS.md` for full machine-specific config

## Compute Equivalence

The competition runs on **8x H100 SXM for 10 min**. Prior testing showed:

| Setup | Approx tokens/sec | ~Equiv wallclock for 816M tokens |
|-------|-------------------|----------------------------------|
| 8x H100 (competition) | ~1.35M | 10 min |
| Desktop 4090 eGPU (24 GB) | ~200K | **~70 min** |
| Laptop 4090 (16 GB) | ~168K | ~80 min |

So to roughly match the competition's token count (~816M), we need **~70 min
on the desktop 4090**. A 20-min budget gives only ~240M tokens (~30% of
competition). The goal here is not to match absolute BPB but to establish
**relative rankings** and measure **throughput** on our hardware.

### Multi-checkpoint recording

Rather than running separate experiments at different budgets, we run a single
**20-min** experiment with frequent validation and record val_bpb at multiple
wallclock checkpoints: **5 / 10 / 15 / 20 min**. This gives a learning curve
for each submission at the cost of one run.

The log lines include `train_time:<ms>`, so we post-process to extract the
val_bpb closest to each checkpoint:

| Checkpoint | train_time (ms) | Tokens (approx) | Notes |
|------------|----------------|-----------------|-------|
| 5 min | 300,000 | ~60M | Small model quick screening |
| 10 min | 600,000 | ~120M | Competition wallclock (but 1/7 the compute) |
| 15 min | 900,000 | ~180M | Diminishing returns test |
| 20 min | 1,200,000 | ~240M | Full benchmark budget |

To get readings near each checkpoint, set `VAL_LOSS_EVERY` appropriately.
**Important**: on 1x 4090 with 8x grad accumulation, each optimizer step takes
~3.1s (vs ~43ms on 8xH100). The default `VAL_LOSS_EVERY=1000` would mean
validation every ~52 min — way past our budget. Override to `VAL_LOSS_EVERY=50`
for ~2.6 min granularity (~8 checkpoints in 20 min).

## Benchmark Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Wallclock | 20 min (1200s) | Single run per submission |
| VAL_LOSS_EVERY | 50 | ~2.5 min granularity for multi-checkpoint extraction |
| GPU count | 1 | Single 4090 |
| Artifact limit | 16 MB | Same as competition |
| Dataset | FineWeb-10B SP1024 | Same as competition |
| Metric | val_bpb at 5/10/15/20 min checkpoints | Learning curve per submission |

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

For each submission in `records/track_10min_16mb/<submission>/`:

```bash
# Copy the submission's train_gpt.py to a temp location
cp records/track_10min_16mb/<submission>/train_gpt.py ./bench_train.py

# Run 20 min with frequent validation for multi-checkpoint extraction
# (See AGENTS.md for machine-specific env vars: CUDA_VISIBLE_DEVICES, LD_LIBRARY_PATH, etc.)
TORCHINDUCTOR_MIX_ORDER_REDUCTION=0 \
MAX_WALLCLOCK_SECONDS=1200 \
VAL_LOSS_EVERY=50 \
python bench_train.py > bench.log 2>&1

# Extract all val_bpb readings with timestamps
# Format: step:100/20000 val_loss:2.5687 val_bpb:1.5174 train_time:310200ms step_avg:3102.00ms
grep "step:.*val_bpb:" bench.log

# Find readings closest to 5/10/15/20 min marks
# (train_time in ms: 300000=5min, 600000=10min, 900000=15min, 1200000=20min)
# Manually find the line with train_time closest to each target

# Peak memory (in MiB)
grep "peak memory allocated:" bench.log

# Final post-quant metric
grep "final_int8_zlib_roundtrip " bench.log
```

**Important notes:**
- Some submissions use `zstandard` — install with `pip install zstandard` if needed
- Some submissions may not have `MAX_WALLCLOCK_SECONDS` — check how wallclock is controlled (some use `ITERATIONS` instead; set both)
- DDP code should gracefully handle single-GPU (WORLD_SIZE=1)
- If a submission OOMs on 24 GB, reduce batch size via env vars (e.g. `TRAIN_BATCH_TOKENS=262144`)
- Override `VAL_LOSS_EVERY=50` to ensure enough checkpoints for multi-point extraction

## Expected Differences from Competition

1. **Fewer tokens**: ~240M in 20 min vs ~816M in competition (10 min × 8 GPUs). We see ~30% of the tokens — expect ~0.03-0.08 BPB worse.
2. **Different throughput profile**: 4090 has higher memory bandwidth per FLOP than H100 SXM. Bandwidth-bound ops (attention) may be relatively faster.
3. **No DDP overhead**: Single GPU avoids all-reduce communication cost, so each step is slightly more efficient per-GPU.
4. **Triton differences**: 4090 (SM89) has 99 KB SMEM vs H100's 228 KB. The `TORCHINDUCTOR_MIX_ORDER_REDUCTION=0` fix is required.
5. **Grad accumulation**: With 1 GPU, scripts that assumed 8-way DDP will use 8x grad accumulation. Same effective batch but more micro-steps per update.
6. **Warmdown schedule**: With `MAX_WALLCLOCK_SECONDS=1200`, warmdown starts at a different absolute step count. Submissions with `warmdown_iters` tuned for 10-min H100 runs may behave differently.

## Results: results_benchmark.tsv

Log results to `results_benchmark.tsv` (tab-separated). Do not commit this file.

```
submission	bpb_5m	bpb_10m	bpb_15m	bpb_20m	bpb_final_quant	memory_gb	tokens_M	status	notes
```

Columns:
1. `submission` — directory name (e.g. `2026-03-20_10L_Int5MLP_MuonWD04_SWA50`)
2. `bpb_5m` — val_bpb at ~5 min (nearest checkpoint to train_time:300000ms)
3. `bpb_10m` — val_bpb at ~10 min
4. `bpb_15m` — val_bpb at ~15 min
5. `bpb_20m` — val_bpb at ~20 min (last checkpoint before wallclock cap)
6. `bpb_final_quant` — post int8+zlib roundtrip val_bpb (final metric)
7. `memory_gb` — peak VRAM in GB
8. `tokens_M` — total tokens processed (millions)
9. `status` — `ok`, `oom`, `crash`, `timeout`
10. `notes` — any issues encountered

Use `—` for missing checkpoints (e.g. if a run crashes before 15 min).

Example:
```
submission	bpb_5m	bpb_10m	bpb_15m	bpb_20m	bpb_final_quant	memory_gb	tokens_M	status	notes
NaiveBaseline	1.3200	1.2800	1.2600	1.2500	1.2550	8.5	240	ok	reference baseline
10L_Int5MLP_MuonWD04_SWA50	1.2900	1.2400	1.2200	1.2100	1.2150	12.3	210	ok	best competition submission
SomeCrash	1.3500	—	—	—	—	15.0	30	crash	OOM at step 120
```

## Submissions to Benchmark

Priority order (by competition val_bpb, best first):

| # | Submission | Competition BPB | Key Techniques |
|---|-----------|----------------|----------------|
| 1 | `NaiveBaseline` | 1.2244 | Reference baseline — run first |
| 2 | `LongContextSeq2048` | 1.2058 | seq=2048, lower LR |
| 3 | `TrainingOptSeq4096` | 1.2014 | seq=4096, Muon tuning |
| 4 | `SlidingWindowEval` | 1.1925 | Sliding window eval only |
| 5 | `SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit` | 1.1748 | 10L, overtone init |
| 6 | `MixedQuant_Int6Int8_SlidingWindow` | 1.1630 | MLP 3x, mixed quant |
| 7 | `Seq2048_FP16Emb_TunedLR` | 1.1598 | 10L, QAT int6, seq=2048 |
| 8 | `smeargate_orthoinit_muonwd` | 1.1556 | SmearGate, BigramHash |
| 9 | `MLP3x_QAT_Int6_SlidingWindow` | 1.1502 | 11L, int6 QAT, zstd-22 |
| 10 | `Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA` | 1.1458 | 7 stacked techniques |
| 11 | `10L_Int5MLP_MuonWD04_SWA50` | 1.1428 | Best: int5/int6, 10L, SWA |

Skip these (less interesting or incomplete):
- `LowerLR` — pure LR sweep, subsumed by others
- `FP16Embed_WD3600` — subsumed by later submissions
- `10L_MixedPrecision` — subsumed by later submissions
- `WarmdownQuantization` — interesting idea but subsumed
- `int6_STE QAT_ MLP_bigram _U_Net` — incomplete (no train_gpt.py)

## Analysis

After running all benchmarks, compare:

1. **Learning curves**: Plot bpb_5m / bpb_10m / bpb_15m / bpb_20m for each submission. Which techniques improve fastest? Which need more tokens to shine?
2. **Relative ranking at each checkpoint**: Does the ranking at 5 min predict 20 min? If so, 5-min screening is viable for autoresearch.
3. **Absolute BPB on 4090 vs competition**: How much does reduced compute hurt? (Expected: ~0.03-0.08 BPB worse at 20 min)
4. **Throughput**: Which techniques hurt tokens/sec the most on 4090? Compare `tokens_M` across submissions.
5. **VRAM**: Which techniques fit comfortably in 24 GB vs needing adjustment?
6. **Scaling efficiency**: Do techniques that need many tokens (e.g. SWA) show steeper improvement curves from 5→20 min?
7. **Quantization gap**: Compare `bpb_20m` (pre-quant) vs `bpb_final_quant`. Which techniques are most robust to quantization?
