import importlib
import os
import sys
import unittest
from contextlib import nullcontext
from unittest import mock

import torch


def import_train_xt_mac():
    sys.modules.pop("train_xt_mac", None)
    return importlib.import_module("train_xt_mac")


class _FakeSentencePieceProcessor:
    def load(self, path):
        self.path = path
        return True

    def vocab_size(self):
        return 1024


class _FakeDecoder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeTransformerWrapper(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        self.weight = torch.nn.Parameter(torch.zeros(()))

    def forward(self, input_ids):
        batch, seq_len = input_ids.shape
        vocab_size = self.kwargs.get("num_tokens", 1024)
        return torch.zeros((batch, seq_len, vocab_size), dtype=torch.float32, device=input_ids.device)


class _FakeXTModel(torch.nn.Module):
    def __init__(self, model, logit_softcap=30.0):
        super().__init__()
        del model, logit_softcap
        self.weight = torch.nn.Parameter(torch.zeros(()))

    def forward(self, input_ids, target_ids):
        del input_ids, target_ids
        return self.weight * 0 + 2.5


class _FakeLoader:
    def __init__(self, pattern, rank, world_size, device):
        del pattern, rank, world_size
        self.device = device

    def next_batch(self, global_tokens, seq_len, grad_accum_steps):
        del global_tokens, grad_accum_steps
        x = torch.zeros((1, seq_len), dtype=torch.int64, device=self.device)
        y = torch.zeros((1, seq_len), dtype=torch.int64, device=self.device)
        return x, y


class _FakeOptimizer:
    def __init__(self, params, **kwargs):
        del params, kwargs
        self.param_groups = [{"lr": 0.0}]

    def zero_grad(self, set_to_none=True):
        del set_to_none

    def step(self):
        return None


def fake_training_components():
    return {
        "XTModel": _FakeXTModel,
        "DistributedTokenLoader": _FakeLoader,
        "build_sentencepiece_luts": lambda sp, vocab_size, device: (
            torch.ones(vocab_size, dtype=torch.int16, device=device),
            torch.zeros(vocab_size, dtype=torch.bool, device=device),
            torch.ones(vocab_size, dtype=torch.bool, device=device),
        ),
        "dequantize_state_dict_int8": lambda obj: obj,
        "load_validation_tokens": lambda pattern, seq_len: torch.zeros(seq_len * 40 + 1, dtype=torch.int64),
        "quantize_state_dict_int8": lambda state_dict: ({}, {"baseline_tensor_bytes": 1, "int8_payload_bytes": 1}),
        "spm": type("FakeSPM", (), {"SentencePieceProcessor": _FakeSentencePieceProcessor}),
        "TransformerWrapper": _FakeTransformerWrapper,
        "Decoder": _FakeDecoder,
    }


class TrainXtMacHelperTests(unittest.TestCase):
    def test_defaults_micro_batch_seqs_to_four_when_possible(self):
        module = import_train_xt_mac()

        micro_batch_seqs, grad_accum_steps = module.resolve_microbatch_settings(
            train_batch_tokens=8192,
            train_seq_len=1024,
            micro_batch_seqs=None,
        )

        self.assertEqual(micro_batch_seqs, 4)
        self.assertEqual(grad_accum_steps, 2)

    def test_import_succeeds_without_dataset_files(self):
        with mock.patch.dict(
            os.environ,
            {
                "DATA_PATH": "/definitely/missing/data",
                "TOKENIZER_PATH": "/definitely/missing/tokenizer.model",
            },
            clear=False,
        ):
            module = import_train_xt_mac()

        self.assertTrue(hasattr(module, "load_training_components"))

    def test_load_training_components_is_lazy(self):
        module = import_train_xt_mac()

        self.assertTrue(callable(module.load_training_components))

    def test_format_val_metrics_log_produces_parseable_numeric_line(self):
        module = import_train_xt_mac()

        line = module.format_val_metrics_log(step=3, iterations=50, val_loss=2.5, val_bpb=1.25, train_time_ms=123)

        self.assertEqual(
            line,
            "step:3/50 val_loss:2.5000 val_bpb:1.2500 train_time:123ms",
        )

    def test_metadata_prefix_does_not_use_val_bpb(self):
        module = import_train_xt_mac()

        line = module.format_tokenizer_metadata_log("/tmp/tokenizer with spaces.model")

        self.assertTrue(line.startswith(module.TOKENIZER_METADATA_PREFIX))
        self.assertNotIn("val_bpb:", line)
        self.assertIn("'/tmp/tokenizer with spaces.model'", line)

    def test_artifact_export_is_disabled_by_default(self):
        module = import_train_xt_mac()

        self.assertFalse(module.should_export_artifacts(None))

    def test_artifact_export_honors_truthy_env_value(self):
        module = import_train_xt_mac()

        self.assertTrue(module.should_export_artifacts("1"))

    def test_from_env_reads_values_at_runtime(self):
        with mock.patch.dict(os.environ, {"RUN_ID": "runtime-run", "EXPORT_ARTIFACTS": "1"}, clear=False):
            module = import_train_xt_mac()
            args = module.Hyperparameters.from_env()

        self.assertEqual(args.run_id, "runtime-run")
        self.assertTrue(args.export_artifacts)

    def test_from_env_defaults_to_smoke_friendly_val_cap(self):
        module = import_train_xt_mac()

        args = module.Hyperparameters.from_env({})

        self.assertGreater(args.val_max_tokens, 0)

    def test_from_env_allows_disabling_val_trim(self):
        module = import_train_xt_mac()

        args = module.Hyperparameters.from_env({"VAL_MAX_TOKENS": "0"})

        self.assertEqual(args.val_max_tokens, 0)

    def test_rejects_distributed_env_for_single_process_script(self):
        module = import_train_xt_mac()

        with self.assertRaisesRegex(RuntimeError, "single-process"):
            module.ensure_single_process_env({"WORLD_SIZE": "2"})

        with self.assertRaisesRegex(RuntimeError, "train_xt_mac.py"):
            module.ensure_single_process_env({"WORLD_SIZE": "2"})

    def test_main_wires_compile_and_logs_numeric_val_bpb(self):
        module = import_train_xt_mac()
        log_sink = mock.mock_open()

        with mock.patch.dict(
            os.environ,
            {
                "RUN_ID": "xt-mac-test",
                "FORCE_CPU": "1",
                "TORCH_COMPILE": "1",
                "ITERATIONS": "0",
                "VAL_LOSS_EVERY": "1",
                "TRAIN_BATCH_TOKENS": "4096",
                "VAL_BATCH_SIZE": "4096",
            },
            clear=False,
        ), mock.patch.object(module, "load_training_components", return_value=fake_training_components()), mock.patch.object(
            module, "get_device", return_value=torch.device("cpu")
        ), mock.patch.object(module, "maybe_compile", side_effect=lambda model, **kwargs: model) as compile_mock, mock.patch.object(
            module.Path, "read_text", return_value="val_bpb: source noise"
        ), mock.patch.object(module.os, "makedirs"), mock.patch.object(module.torch.optim, "AdamW", _FakeOptimizer), mock.patch(
            "builtins.open", log_sink
        ):
            module.main()

        compile_mock.assert_called_once()
        self.assertEqual(compile_mock.call_args.kwargs["device_type"], "cpu")
        self.assertTrue(compile_mock.call_args.kwargs["requested"])

        written = "".join(call.args[0] for call in log_sink().write.call_args_list)
        self.assertIn("step:0/0 val_loss:2.5000 val_bpb:", written)
        self.assertIn("artifact_export:disabled", written)
        self.assertIn(f"val_loader:shards pattern=./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin tokens:{module.Hyperparameters().val_max_tokens}", written)
        self.assertEqual(written.count("val_bpb:"), 1)

    def test_main_uses_eval_context_helper_for_validation(self):
        module = import_train_xt_mac()

        with mock.patch.dict(
            os.environ,
            {
                "RUN_ID": "xt-mac-test",
                "FORCE_CPU": "1",
                "ITERATIONS": "0",
                "VAL_LOSS_EVERY": "1",
                "TRAIN_BATCH_TOKENS": "4096",
                "VAL_BATCH_SIZE": "4096",
            },
            clear=False,
        ), mock.patch.object(module, "load_training_components", return_value=fake_training_components()), mock.patch.object(
            module, "get_device", return_value=torch.device("cpu")
        ), mock.patch.object(module, "get_eval_context", return_value=nullcontext(), create=True) as eval_context_mock, mock.patch.object(
            module.Path, "read_text", return_value="val_bpb: source noise"
        ), mock.patch.object(module.os, "makedirs"), mock.patch.object(module.torch.optim, "AdamW", _FakeOptimizer), mock.patch(
            "builtins.open", mock.mock_open()
        ):
            module.main()

        eval_context_mock.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
