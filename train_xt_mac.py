from __future__ import annotations

import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast

import torch

from macos_torch_runtime import (
    DEFAULT_MACOS_VAL_MAX_TOKENS,
    adam_optimizer_kwargs,
    get_autocast_context,
    get_device,
    get_eval_context,
    metric_accum_dtype,
    maybe_compile,
    peak_memory_summary,
    synchronize_device,
    trim_validation_tokens_for_smoke,
)


@dataclass(slots=True)
class Hyperparameters:
    data_path: str = "./data/datasets/fineweb10B_sp1024"
    tokenizer_path: str = "./data/tokenizers/fineweb_1024_bpe.model"
    run_id: str = "xt-mac-local"
    seed: int = 1337

    val_batch_size: int = 4096
    val_max_tokens: int = DEFAULT_MACOS_VAL_MAX_TOKENS
    val_loss_every: int = 25
    train_log_every: int = 10

    iterations: int = 50
    warmdown_iters: int = 10
    train_batch_tokens: int = 4096
    train_seq_len: int = 1024
    max_wallclock_seconds: float = 30.0

    vocab_size: int = 1024
    num_layers: int = 4
    model_dim: int = 256
    num_heads: int = 4
    logit_softcap: float = 30.0

    use_rmsnorm: bool = True
    attn_flash: bool = False
    attn_qk_norm: bool = True
    rotary_pos_emb: bool = True
    ff_swish: bool = True
    ff_glu: bool = True
    shift_tokens: int = 0
    add_value_residual: bool = True
    softclamp_output: bool = True
    zero_init_branch_output: bool = True

    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    adam_eps: float = 1e-8
    grad_clip_norm: float = 0.0
    micro_batch_seqs: int | None = None

    force_cpu: bool = False
    use_torch_compile: bool = False
    export_artifacts: bool = False

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Hyperparameters":
        env_map: Mapping[str, str] = os.environ if env is None else env
        defaults = cls()
        return cls(
            data_path=env_map.get("DATA_PATH", defaults.data_path),
            tokenizer_path=env_map.get("TOKENIZER_PATH", defaults.tokenizer_path),
            run_id=env_map.get("RUN_ID", defaults.run_id),
            seed=int(env_map.get("SEED", str(defaults.seed))),
            val_batch_size=int(env_map.get("VAL_BATCH_SIZE", str(defaults.val_batch_size))),
            val_max_tokens=int(env_map.get("VAL_MAX_TOKENS", str(defaults.val_max_tokens))),
            val_loss_every=int(env_map.get("VAL_LOSS_EVERY", str(defaults.val_loss_every))),
            train_log_every=int(env_map.get("TRAIN_LOG_EVERY", str(defaults.train_log_every))),
            iterations=int(env_map.get("ITERATIONS", str(defaults.iterations))),
            warmdown_iters=int(env_map.get("WARMDOWN_ITERS", str(defaults.warmdown_iters))),
            train_batch_tokens=int(env_map.get("TRAIN_BATCH_TOKENS", str(defaults.train_batch_tokens))),
            train_seq_len=int(env_map.get("TRAIN_SEQ_LEN", str(defaults.train_seq_len))),
            max_wallclock_seconds=float(env_map.get("MAX_WALLCLOCK_SECONDS", str(defaults.max_wallclock_seconds))),
            vocab_size=int(env_map.get("VOCAB_SIZE", str(defaults.vocab_size))),
            num_layers=int(env_map.get("NUM_LAYERS", str(defaults.num_layers))),
            model_dim=int(env_map.get("MODEL_DIM", str(defaults.model_dim))),
            num_heads=int(env_map.get("NUM_HEADS", str(defaults.num_heads))),
            logit_softcap=float(env_map.get("LOGIT_SOFTCAP", str(defaults.logit_softcap))),
            use_rmsnorm=bool(int(env_map.get("USE_RMSNORM", "1"))),
            attn_flash=bool(int(env_map.get("ATTN_FLASH", "0"))),
            attn_qk_norm=bool(int(env_map.get("ATTN_QK_NORM", "1"))),
            rotary_pos_emb=bool(int(env_map.get("ROTARY_POS_EMB", "1"))),
            ff_swish=bool(int(env_map.get("FF_SWISH", "1"))),
            ff_glu=bool(int(env_map.get("FF_GLU", "1"))),
            shift_tokens=int(env_map.get("SHIFT_TOKENS", str(defaults.shift_tokens))),
            add_value_residual=bool(int(env_map.get("ADD_VALUE_RESIDUAL", "1"))),
            softclamp_output=bool(int(env_map.get("SOFTCLAMP_OUTPUT", "1"))),
            zero_init_branch_output=bool(int(env_map.get("ZERO_INIT_BRANCH_OUTPUT", "1"))),
            learning_rate=float(env_map.get("LEARNING_RATE", str(defaults.learning_rate))),
            weight_decay=float(env_map.get("WEIGHT_DECAY", str(defaults.weight_decay))),
            beta1=float(env_map.get("BETA1", str(defaults.beta1))),
            beta2=float(env_map.get("BETA2", str(defaults.beta2))),
            adam_eps=float(env_map.get("ADAM_EPS", str(defaults.adam_eps))),
            grad_clip_norm=float(env_map.get("GRAD_CLIP_NORM", str(defaults.grad_clip_norm))),
            micro_batch_seqs=int(env_map.get("MICRO_BATCH_SEQS", "0")) or None,
            force_cpu=bool(int(env_map.get("FORCE_CPU", "0"))),
            use_torch_compile=bool(int(env_map.get("TORCH_COMPILE", "0"))),
            export_artifacts=should_export_artifacts(env_map.get("EXPORT_ARTIFACTS")),
        )

    @property
    def train_files(self) -> str:
        return os.path.join(self.data_path, "fineweb_train_*.bin")

    @property
    def val_files(self) -> str:
        return os.path.join(self.data_path, "fineweb_val_*.bin")


def resolve_microbatch_settings(
    train_batch_tokens: int,
    train_seq_len: int,
    micro_batch_seqs: int | None,
) -> tuple[int, int]:
    if train_batch_tokens <= 0:
        raise ValueError("TRAIN_BATCH_TOKENS must be positive")
    if train_seq_len <= 0:
        raise ValueError("TRAIN_SEQ_LEN must be positive")
    if train_batch_tokens % train_seq_len != 0:
        raise ValueError("TRAIN_BATCH_TOKENS must be divisible by TRAIN_SEQ_LEN")

    total_train_seqs = train_batch_tokens // train_seq_len
    resolved_micro_batch_seqs = min(total_train_seqs, 4) if micro_batch_seqs is None else micro_batch_seqs
    if resolved_micro_batch_seqs <= 0:
        raise ValueError("MICRO_BATCH_SEQS must be positive")
    if total_train_seqs % resolved_micro_batch_seqs != 0:
        raise ValueError(
            "MICRO_BATCH_SEQS must divide TRAIN_BATCH_TOKENS / TRAIN_SEQ_LEN "
            f"(got {resolved_micro_batch_seqs} for {total_train_seqs} total sequences)"
        )
    return resolved_micro_batch_seqs, total_train_seqs // resolved_micro_batch_seqs


def should_export_artifacts(raw_value: str | None) -> bool:
    return str(raw_value or "0").strip().lower() in {"1", "true", "yes", "on"}


TOKENIZER_METADATA_PREFIX = "tokenizer_metadata:"


def format_tokenizer_metadata_log(tokenizer_path: str) -> str:
    return f"{TOKENIZER_METADATA_PREFIX} tokenizer_kind=sentencepiece tokenizer_path={tokenizer_path!r}"


def ensure_single_process_env(env: Mapping[str, str] | None = None) -> None:
    env_map: Mapping[str, str] = os.environ if env is None else env
    distributed_keys = ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")
    if any(key in env_map for key in distributed_keys):
        present = ", ".join(key for key in distributed_keys if key in env_map)
        raise RuntimeError(f"train_xt_mac.py is single-process only; found distributed env: {present}")


def format_val_metrics_log(step: int, iterations: int, val_loss: float, val_bpb: float, train_time_ms: int) -> str:
    return f"step:{step}/{iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} train_time:{train_time_ms}ms"


def load_training_components() -> dict[str, Any]:
    import sentencepiece as spm

    from train_xt import (
        Decoder,
        DistributedTokenLoader,
        TransformerWrapper,
        XTModel,
        build_sentencepiece_luts,
        dequantize_state_dict_int8,
        load_validation_tokens,
        quantize_state_dict_int8,
    )

    return {
        "Decoder": Decoder,
        "DistributedTokenLoader": DistributedTokenLoader,
        "TransformerWrapper": TransformerWrapper,
        "XTModel": XTModel,
        "build_sentencepiece_luts": build_sentencepiece_luts,
        "dequantize_state_dict_int8": dequantize_state_dict_int8,
        "load_validation_tokens": load_validation_tokens,
        "quantize_state_dict_int8": quantize_state_dict_int8,
        "spm": spm,
    }


def maybe_export_artifacts(args: Hyperparameters, base_model: torch.nn.Module, code: str, eval_val_fn, log0) -> None:
    if not args.export_artifacts:
        log0("artifact_export:disabled")
        return

    import io
    import zlib

    components = load_training_components()
    quantize_state_dict_int8 = components["quantize_state_dict_int8"]
    dequantize_state_dict_int8 = components["dequantize_state_dict_int8"]

    torch.save(base_model.state_dict(), "final_xt_model_mac.pt")
    model_bytes = os.path.getsize("final_xt_model_mac.pt")
    code_bytes = len(code.encode("utf-8"))
    log0(f"Serialized model: {model_bytes} bytes")
    log0(f"Code size: {code_bytes} bytes")
    log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_raw_bytes = len(quant_raw)
    with open("final_xt_model_mac.int8.ptz", "wb") as f:
        f.write(quant_blob)
    quant_file_bytes = os.path.getsize("final_xt_model_mac.int8.ptz")
    ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
    log0(
        f"Serialized model int8+zlib: {quant_file_bytes} bytes "
        f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
    )
    log0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")

    with open("final_xt_model_mac.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    q_val_loss, q_val_bpb = eval_val_fn(base_model)
    log0(f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f}")
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")


def main() -> None:
    ensure_single_process_env()
    components = load_training_components()
    spm = components["spm"]
    XTModel = components["XTModel"]
    Decoder = components["Decoder"]
    TransformerWrapper = components["TransformerWrapper"]
    DistributedTokenLoader = components["DistributedTokenLoader"]
    load_validation_tokens = components["load_validation_tokens"]
    build_sentencepiece_luts = components["build_sentencepiece_luts"]

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters.from_env()
    device = get_device(force_cpu=args.force_cpu)
    micro_batch_seqs, grad_accum_steps = resolve_microbatch_settings(
        train_batch_tokens=args.train_batch_tokens,
        train_seq_len=args.train_seq_len,
        micro_batch_seqs=args.micro_batch_seqs,
    )
    grad_scale = 1.0 / grad_accum_steps

    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{args.run_id}.txt"
    print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if console:
            print(msg)
        with open(logfile, "a", encoding="utf-8") as f:
            print(msg, file=f)

    log0(f"source_file:{Path(__file__).name} source_bytes:{len(code.encode('utf-8'))}", console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0("=" * 100, console=False)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only supports SentencePiece .model files: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )

    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = trim_validation_tokens_for_smoke(
        load_validation_tokens(args.val_files, args.train_seq_len),
        seq_len=args.train_seq_len,
        val_max_tokens=args.val_max_tokens,
    )
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    metric_dtype = metric_accum_dtype(device.type)
    log0(f"device:{device.type} compile_requested:{args.use_torch_compile}")
    log0(format_tokenizer_metadata_log(args.tokenizer_path))
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    decoder_kwargs = dict(
        dim=args.model_dim,
        depth=args.num_layers,
        heads=args.num_heads,
        rotary_pos_emb=args.rotary_pos_emb,
        attn_flash=args.attn_flash,
        attn_qk_norm=args.attn_qk_norm,
        use_rmsnorm=args.use_rmsnorm,
        ff_swish=args.ff_swish,
        ff_glu=args.ff_glu,
        shift_tokens=args.shift_tokens,
        add_value_residual=args.add_value_residual,
        softclamp_output=args.softclamp_output,
        zero_init_branch_output=args.zero_init_branch_output,
    )
    xt_model = TransformerWrapper(
        num_tokens=args.vocab_size,
        max_seq_len=args.train_seq_len,
        attn_layers=Decoder(**decoder_kwargs),
    )
    base_model = XTModel(xt_model, logit_softcap=args.logit_softcap).to(device)
    model = cast(torch.nn.Module, maybe_compile(base_model, device_type=device.type, requested=args.use_torch_compile, log_fn=log0))

    optimizer_kwargs = cast(dict[str, bool], adam_optimizer_kwargs(device.type))
    optimizer = torch.optim.AdamW(
        base_model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
        **optimizer_kwargs,
    )
    train_loader = DistributedTokenLoader(args.train_files, rank=0, world_size=1, device=device)

    log0(f"model_params:{sum(p.numel() for p in base_model.parameters())}")
    log0(f"grad_accum_steps:{grad_accum_steps} micro_batch_seqs:{micro_batch_seqs}")
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            if warmdown_start <= step < args.iterations:
                return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
            return 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    def eval_val_local(eval_model: torch.nn.Module) -> tuple[float, float]:
        local_batch_tokens = args.val_batch_size // grad_accum_steps
        if local_batch_tokens < args.train_seq_len:
            raise ValueError(
                "VAL_BATCH_SIZE must provide at least one sequence per accumulation step; "
                f"got VAL_BATCH_SIZE={args.val_batch_size}, GRAD_ACCUM_STEPS={grad_accum_steps}, "
                f"TRAIN_SEQ_LEN={args.train_seq_len}"
            )
        local_batch_seqs = local_batch_tokens // args.train_seq_len
        total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
        val_loss_sum = 0.0
        val_token_count = 0.0
        val_byte_count = 0.0

        eval_model.eval()
        with get_eval_context():
            for batch_seq_start in range(0, total_seqs, local_batch_seqs):
                batch_seq_end = min(batch_seq_start + local_batch_seqs, total_seqs)
                raw_start = batch_seq_start * args.train_seq_len
                raw_end = batch_seq_end * args.train_seq_len + 1
                local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64)
                x = local[:-1].reshape(-1, args.train_seq_len)
                y = local[1:].reshape(-1, args.train_seq_len)
                with get_autocast_context(device.type, enabled=True):
                    batch_loss = eval_model(x, y).detach()
                batch_token_count = float(y.numel())
                val_loss_sum += float(batch_loss.item()) * batch_token_count
                val_token_count += batch_token_count
                prev_ids = x.reshape(-1)
                tgt_ids = y.reshape(-1)
                token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
                token_bytes += (
                    has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
                ).to(dtype=torch.int16)
                val_byte_count += float(token_bytes.to(metric_dtype).sum().item())

        val_loss = val_loss_sum / val_token_count
        bits_per_token = val_loss / math.log(2.0)
        tokens_per_byte = val_token_count / val_byte_count
        eval_model.train()
        return val_loss, bits_per_token * tokens_per_byte

    training_time_ms = 0.0
    stop_after_step = None
    synchronize_device(device.type)
    t0 = time.perf_counter()
    step = 0

    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            synchronize_device(device.type)
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val_local(model)
            log0(
                format_val_metrics_log(
                    step=step,
                    iterations=args.iterations,
                    val_loss=val_loss,
                    val_bpb=val_bpb,
                    train_time_ms=int(training_time_ms),
                )
            )
            synchronize_device(device.type)
            t0 = time.perf_counter()

        if last_step:
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        optimizer.zero_grad(set_to_none=True)
        train_loss = torch.zeros((), device=device)
        for _ in range(grad_accum_steps):
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with get_autocast_context(device.type, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * scale
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None):
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        if stop_after_step is None and max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms:
            stop_after_step = step

    log0(peak_memory_summary(device.type))
    maybe_export_artifacts(args, base_model, code, eval_val_local, log0)


if __name__ == "__main__":
    main()
