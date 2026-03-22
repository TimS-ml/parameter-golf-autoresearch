# parameter-golf: Competition Technique Porting

This program guides porting competition techniques from `records/` submissions
into our codebase (`train_gpt.py` or `train_xt.py`), testing them on 1x RTX 4090.

## Goal

The `records/track_10min_16mb/` directory contains 17 submissions from the
parameter-golf competition (10 min on 8xH100, 16MB artifact limit). Many use
custom code changes not available in x-transformers. We want to:

1. **Understand** each technique's contribution via ablation
2. **Port** the most impactful techniques into our codebase
3. **Validate** that they work on 1x RTX 4090 with 20-min wallclock
4. **Measure** throughput impact (tokens/sec) on our hardware

## Environment

See `AGENTS.md` for hardware details. Key differences from the competition:

| Aspect | Competition | Our Setup |
|--------|------------|-----------|
| GPU | 8x H100 SXM (80 GB each) | 1x RTX 4090 eGPU (24 GB) |
| Wallclock | 10 min | 20 min (autoresearch) / 70 min (benchmark parity) |
| Tokens/sec | ~1.35M (8 GPUs) | ~200K (1 GPU) |
| DDP | 8-way | Single GPU (8x grad accum) |
| Effective batch | 524K tokens / step | Same (via grad accumulation) |

### Launch command template

```bash
# See AGENTS.md for machine-specific env vars (CUDA_VISIBLE_DEVICES, LD_LIBRARY_PATH, python path)
TORCHINDUCTOR_MIX_ORDER_REDUCTION=0 \
MAX_WALLCLOCK_SECONDS=1200 \
python train_gpt.py > run.log 2>&1
```

## Technique Catalog

Techniques from competition records, ordered by BPB impact:

### Tier 1: High Impact (> 0.01 BPB)

#### 1. Sliding Window Eval (stride=64)
- **Source**: `SlidingWindowEval/`, `MixedQuant_Int6Int8_SlidingWindow/`
- **Impact**: -0.032 BPB (eval-time only, zero training cost)
- **How**: Each token gets ~960 tokens of prior context instead of just its position in a 1024-length chunk. Evaluate overlapping windows with stride=64, only score the last 64 tokens of each window.
- **Port difficulty**: Medium. Eval code change only.
- **Status**: Not ported

#### 2. MLP 3x Expansion
- **Source**: `MixedQuant_Int6Int8_SlidingWindow/`, `smeargate_orthoinit_muonwd/`
- **Impact**: -0.029 BPB
- **How**: `mlp_mult=3` (hidden=1536 vs default 1024). More capacity in FFN.
- **Port difficulty**: Easy. Config change in train_gpt.py, or `ff_mult=6` with `ff_glu=True` in x-transformers (GLU halves effective mult).
- **Status**: Not ported

#### 3. Longer Sequences (2048)
- **Source**: `LongContextSeq2048/`, `Seq2048_FP16Emb_TunedLR/`
- **Impact**: -0.019 BPB
- **How**: `train_seq_len=2048`. Needs LR reduction (~0.02-0.032).
- **Port difficulty**: Easy. Config change.
- **Status**: Not ported

#### 4. Int6 QAT (STE)
- **Source**: `MLP3x_QAT_Int6_SlidingWindow/`, `Seq2048_FP16Emb_TunedLR/`
- **Impact**: -0.010 BPB (eliminates quantization gap entirely)
- **How**: Straight-Through Estimator fake-quantization during training. Quantize weights to int6 [-32,31] in forward pass, pass gradients through as if unquantized.
- **Port difficulty**: Hard. Custom quantization code in forward pass + modified backward.
- **Status**: Not ported

### Tier 2: Medium Impact (0.003-0.01 BPB)

#### 5. SmearGate
- **Source**: `smeargate_orthoinit_muonwd/`, `Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/`
- **Impact**: ~-0.005 BPB (combined with BigramHash)
- **How**: Learned per-dimension gate blending current + previous token embedding. Gate = sigmoid(learnable_bias), initialized at sigmoid(3.0) ≈ 0.95 (mostly pass-through).
- **Port difficulty**: Medium. Small nn.Module addition.
- **Status**: Not ported

#### 6. BigramHash Embedding
- **Source**: `smeargate_orthoinit_muonwd/`, `10L_Int5MLP_MuonWD04_SWA50/`
- **Impact**: ~-0.003 BPB (on top of SmearGate)
- **How**: 4096-10240 bucket hash table (dim=128, projected to model dim) mapping token pairs via `(prev * 92821 + cur) % num_buckets`. Adds bigram context at embedding level.
- **Port difficulty**: Medium. Small nn.Module addition.
- **Status**: Not ported

#### 7. SWA (Stochastic Weight Averaging)
- **Source**: `Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/`, `10L_Int5MLP_MuonWD04_SWA50/`
- **Impact**: -0.005 BPB
- **How**: Average model weights every 50 steps over last 40-50% of training. Produces smoother weight distributions that quantize better.
- **Port difficulty**: Medium. Training loop change (~20 lines).
- **Status**: Not ported

#### 8. FP16 Embedding Passthrough
- **Source**: `FP16Embed_WD3600/`, `Seq2048_FP16Emb_TunedLR/`
- **Impact**: -0.005 BPB (reduces quant degradation from 0.007 to 0.0005)
- **How**: Store tied embedding in fp16 during quantization instead of int8. Reduce MLP hidden to compensate for size increase.
- **Port difficulty**: Easy-Medium. Quant code change.
- **Status**: Not ported

### Tier 3: Lower Impact (< 0.003 BPB)

#### 9. Orthogonal Weight Init
- **Source**: `smeargate_orthoinit_muonwd/`
- **Impact**: ~small (faster convergence in short budget)
- **How**: `nn.init.orthogonal_` on all CastedLinear weights.
- **Port difficulty**: Easy.
- **Status**: Not ported

#### 10. zstd-22 Compression
- **Source**: `MLP3x_QAT_Int6_SlidingWindow/`, `10L_Int5MLP_MuonWD04_SWA50/`
- **Impact**: Indirect (saves ~1.5MB vs zlib, funds more params under 16MB cap)
- **How**: Replace `zlib.compress(data, 9)` with `zstandard.compress(data, level=22)`.
- **Port difficulty**: Easy. `pip install zstandard`.
- **Status**: Not ported

#### 11. Int5 MLP Quantization
- **Source**: `10L_Int5MLP_MuonWD04_SWA50/`
- **Impact**: Indirect (saves ~1.86MB vs int6, funds a 10th layer)
- **How**: MLP weights use int5 [-16,15] (64 levels). Attention weights keep int6.
- **Port difficulty**: Hard (custom quant code).
- **Status**: Not ported

#### 12. U-Net Skip Connections
- **Source**: `MLP3x_QAT_Int6_SlidingWindow/`, `smeargate_orthoinit_muonwd/`
- **Impact**: Small (enables deeper models)
- **How**: First half of layers = encoder, second half = decoder with skip connections from encoder.
- **Port difficulty**: Medium.
- **Status**: Not ported

### Hyperparameter Findings (from competition)

| Parameter | Competition optimal | Baseline | Notes |
|-----------|-------------------|----------|-------|
| LR (matrix/scalar) | 0.02 | 0.04 | Half the default |
| LR (tied embed) | 0.03-0.04 | 0.05 | Slightly lower |
| Muon momentum | 0.99 (warmup from 0.92) | 0.95 | Higher with warmup |
| Weight decay (Muon) | 0.04 | 0 | Significant regularization |
| Warmdown iters | 3000 | 1200 | Longer cooldown |
| Grad clip norm | 0.3 | 1.0 | Tighter clipping |
| Sequence length | 2048 | 1024 | 2x default |
| Batch tokens | 786K | 524K | 1.5x for seq=2048 |

## Porting Workflow

For each technique:

1. **Read** the source submission's `train_gpt.py` and `README.md`
2. **Understand** the implementation — identify the minimal code change
3. **Port** into our `train_gpt.py` (or `train_xt.py` if applicable)
4. **Test** on 1x 4090 with 20-min wallclock
5. **Log** results to `results_golf.tsv`
6. **Keep or discard** based on val_bpb improvement

### results_golf.tsv Format

```
commit	val_bpb	memory_gb	status	description
```

Same format as `results_xt.tsv`. Do not commit this file.

See also `program_benchmark.md` for running competition submissions unmodified
(benchmarking), vs this file which is about porting techniques into our code.

## Porting Priority

Suggested order for maximum impact-per-effort:

1. Sliding window eval (biggest gain, eval-only change)
2. MLP 3x / `ff_mult=6` (easy config, big gain)
3. seq_len=2048 + LR=0.02 (easy config, big gain)
4. Hyperparameter sweep (LR, momentum, warmdown, WD)
5. FP16 embedding passthrough (medium effort, reliable gain)
6. SWA (medium effort, reliable gain)
7. SmearGate + BigramHash (medium effort, moderate gain)
8. Int6 QAT (hard effort, big gain for quantized metric)
9. Orthogonal init (easy, small gain)
10. zstd-22 (easy, indirect gain via size budget)
