# autoresearch (x-transformers edition)

This is an experiment to have the LLM do its own research on subword language
modeling using [x-transformers](https://github.com/lucidrains/x-transformers).

**Task**: Train a small transformer on FineWeb-10B (SentencePiece BPE, vocab=1024)
and minimize **val_bpb** (bits per byte) within a fixed wallclock budget.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar22`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current HEAD.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `AGENTS.md` — **machine-specific overrides** (Python path, GPU VRAM, precision, etc.). Always read this first and follow its settings. This file is not committed — it is customized per machine.
   - `memory/research_plan_xt.md` — **research context & strategy** (if it exists). Contains prior experiment insights, what's been tried, and what to try next. This file is not committed.
   - `train_xt.py` — the file you modify. Model config, optimizer, training loop.
   - `docs/4090-local-training.md` — 4090-specific torch.compile fixes.
   - `x-transformers/x_transformers/x_transformers.py` — the x-transformers core (reference, do not modify).
4. **Verify data exists**: Check that `./data/datasets/fineweb10B_sp1024/` contains train/val `.bin` shards and `./data/tokenizers/fineweb_1024_bpe.model` exists. If not, run:
   ```bash
   # 20 shards (~2B tokens) — sufficient for runs up to ~70 min on 1x 4090
   python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 20
   ```
5. **Verify dependencies**: Run `python -c "from x_transformers import TransformerWrapper"` to check. If missing deps, install:
   ```bash
   pip install loguru einx ema-pytorch adam-atan2-pytorch sentencepiece
   ```
6. **Initialize results_xt.tsv**: Create `results_xt.tsv` with just the header row. The baseline will be recorded after the first run.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU (or multi-GPU via DDP). The training script runs for a **fixed wallclock budget** (configured via `MAX_WALLCLOCK_SECONDS` in `train_xt.py`; default 600s = 10 minutes). Launch it as:

```bash
python train_xt.py                    # single GPU, BF16
torchrun --nproc_per_node=N train_xt.py  # multi-GPU DDP
```

On RTX 4090, you may need the Triton SMEM fix:
```bash
TORCHINDUCTOR_MIX_ORDER_REDUCTION=0 python train_xt.py
```

**What you CAN do:**
- Modify `train_xt.py` — this is the only file you edit. Everything is fair game: model architecture parameters, x-transformers Decoder options, optimizer, hyperparameters, training loop, batch size, model size, sequence length, etc.

**What you CANNOT do:**
- Modify files inside `x-transformers/`. The library is read-only reference.
- Break the output format (the log must include parseable `val_bpb:` lines).

**The goal is simple: get the lowest val_bpb.** Since the wallclock budget is fixed, you don't need to worry about training time. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the wallclock budget.

**VRAM** is a hard constraint. OOM = crash. Be conservative with batch sizes and model dimensions.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Key Differences from Character-Level (enwik8)

This setup differs from the original character-level autoresearch in important ways:

1. **Metric is BPB, not BPC**: Bits per byte accounts for the tokenizer. Lower is better.
2. **BPE tokenizer (vocab=1024)**: Tokens are subword units, not characters. This changes what architectural features help (e.g. `shift_tokens` was critical for char-level but may not help with BPE).
3. **Much larger dataset**: FineWeb-10B has ~10B tokens. In 10 minutes, you'll see a fraction of an epoch. Overfitting is not a concern; throughput (tokens/sec) matters more.
4. **Quantization matters**: The final artifact uses int8+zlib compression. The roundtrip val_bpb after quantization is what counts.
5. **Sequence length**: Default is 1024 tokens (not 4096 chars). Each token covers more text.
6. **Grad accumulation**: Batch size is specified in total tokens (`train_batch_tokens=524288`), split across micro-batches and grad accumulation steps.

## Output format

The script logs validation metrics during training:

```
step:1000/20000 val_loss:2.3456 val_bpb:1.2345 train_time:60000ms
```

And at the end, it runs an int8 quantization roundtrip:

```
final_int8_zlib_roundtrip val_loss:2.3500 val_bpb:1.2350
```

Extract the key metric from the log file:
```
grep "val_bpb:" run.log | tail -1
```

The final metric of interest is the **roundtrip val_bpb** (after int8+zlib quantization).

## Logging results

When an experiment is done, log it to `results_xt.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	1.234567	8.5	keep	baseline (dim=512 depth=7 heads=8 bf16)
b2c3d4e	1.220000	8.6	keep	add scale_residual=True
c3d4e5f	1.250000	8.5	discard	reduce depth to 4
d4e5f6g	0.000000	0.0	crash	dim=1024 OOM
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar22`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train_xt.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `python train_xt.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "val_bpb:\|peak memory" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in `results_xt.tsv` (do not commit — leave untracked by git)
8. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
9. If val_bpb is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~`MAX_WALLCLOCK_SECONDS` total (+ overhead for startup, compilation, eval, and quantization). If a run exceeds 2x the time budget, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read the x-transformers source for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.
