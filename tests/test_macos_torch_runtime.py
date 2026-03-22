import unittest
from unittest import mock

import torch
import macos_torch_runtime

from macos_torch_runtime import (
    adam_optimizer_kwargs,
    choose_device_type,
    get_autocast_context,
    get_device,
    get_eval_context,
    metric_accum_dtype,
    maybe_compile,
    peak_memory_summary,
    trim_validation_tokens_for_smoke,
    should_enable_compile,
    synchronize_device,
)


class MacOSTorchRuntimeTests(unittest.TestCase):
    def test_choose_device_prefers_mps_when_available(self):
        self.assertEqual(choose_device_type(mps_available=True, force_cpu=False), "mps")

    def test_choose_device_falls_back_to_cpu(self):
        self.assertEqual(choose_device_type(mps_available=False, force_cpu=False), "cpu")

    def test_choose_device_honors_force_cpu(self):
        self.assertEqual(choose_device_type(mps_available=True, force_cpu=True), "cpu")

    def test_get_device_uses_detected_mps_when_available(self):
        with mock.patch("torch.backends.mps.is_available", return_value=True):
            self.assertEqual(get_device().type, "mps")

    def test_get_device_can_force_cpu(self):
        with mock.patch("torch.backends.mps.is_available", return_value=True):
            self.assertEqual(get_device(force_cpu=True).type, "cpu")

    def test_get_autocast_context_is_a_no_op_on_cpu(self):
        context = get_autocast_context("cpu", enabled=True)
        with context as entered:
            self.assertIsNone(entered)

    def test_get_autocast_context_is_a_no_op_on_mps(self):
        context = get_autocast_context("mps", enabled=True)
        with context as entered:
            self.assertIsNone(entered)

    def test_get_autocast_context_uses_torch_autocast_for_enabled_cuda(self):
        sentinel = object()

        with mock.patch.object(macos_torch_runtime.torch, "autocast", return_value=sentinel) as autocast_mock:
            context = get_autocast_context("cuda", enabled=True)

        self.assertIs(context, sentinel)
        autocast_mock.assert_called_once_with(device_type="cuda", enabled=True)

    def test_get_eval_context_returns_no_grad_instead_of_inference_mode(self):
        context = get_eval_context()

        self.assertIsInstance(context, torch.no_grad)
        self.assertNotIsInstance(context, torch.inference_mode)

    def test_compile_disabled_by_default_on_mps(self):
        self.assertFalse(should_enable_compile("mps", requested=True))

    def test_compile_enabled_on_cpu_when_requested(self):
        self.assertTrue(should_enable_compile("cpu", requested=True))

    def test_compile_disabled_when_not_requested(self):
        self.assertFalse(should_enable_compile("cpu", requested=False))

    def test_maybe_compile_returns_original_model_when_disabled(self):
        model = object()
        log = mock.Mock()

        compiled = maybe_compile(model, device_type="mps", requested=True, log_fn=log)

        self.assertIs(compiled, model)
        log.assert_not_called()

    def test_maybe_compile_uses_torch_compile_when_enabled(self):
        model = object()
        compiled_model = object()
        log = mock.Mock()

        with mock.patch("torch.compile", return_value=compiled_model) as compile_mock:
            result = maybe_compile(model, device_type="cpu", requested=True, log_fn=log)

        self.assertIs(result, compiled_model)
        compile_mock.assert_called_once_with(model)
        log.assert_not_called()

    def test_maybe_compile_logs_and_falls_back_on_compile_error(self):
        model = object()
        log = mock.Mock()

        with mock.patch("torch.compile", side_effect=RuntimeError("boom")):
            result = maybe_compile(model, device_type="cpu", requested=True, log_fn=log)

        self.assertIs(result, model)
        log.assert_called_once()
        self.assertIn("torch.compile failed", log.call_args[0][0])

    def test_synchronize_device_calls_mps_sync_for_mps(self):
        with mock.patch.object(macos_torch_runtime.torch.mps, "synchronize") as sync_mock:
            synchronize_device("mps")

        sync_mock.assert_called_once_with()

    def test_synchronize_device_is_no_op_on_cpu(self):
        with mock.patch.object(macos_torch_runtime.torch.mps, "synchronize") as sync_mock:
            synchronize_device("cpu")

        sync_mock.assert_not_called()

    def test_synchronize_device_is_defensive_when_mps_sync_is_missing(self):
        fake_mps = object()

        with mock.patch.object(macos_torch_runtime.torch, "mps", fake_mps):
            synchronize_device("mps")

    def test_peak_memory_summary_marks_mps_as_not_supported(self):
        self.assertIn("not_supported", peak_memory_summary("mps"))

    def test_peak_memory_summary_reports_cpu_prefix(self):
        summary = peak_memory_summary("cpu")
        self.assertTrue(summary.startswith("peak memory allocated:"))

    def test_adam_optimizer_kwargs_avoids_fused_for_cpu(self):
        self.assertEqual(adam_optimizer_kwargs("cpu"), {})

    def test_adam_optimizer_kwargs_avoids_fused_for_mps(self):
        self.assertEqual(adam_optimizer_kwargs("mps"), {})

    def test_adam_optimizer_kwargs_enables_fused_for_cuda(self):
        self.assertEqual(adam_optimizer_kwargs("cuda"), {"fused": True})

    def test_metric_accum_dtype_uses_float32_on_mps(self):
        self.assertIs(metric_accum_dtype("mps"), torch.float32)

    def test_metric_accum_dtype_uses_float64_on_cpu(self):
        self.assertIs(metric_accum_dtype("cpu"), torch.float64)

    def test_metric_accum_dtype_uses_float64_on_cuda(self):
        self.assertIs(metric_accum_dtype("cuda"), torch.float64)

    def test_trim_validation_tokens_for_smoke_keeps_full_validation_when_disabled(self):
        tokens = torch.arange(4097, dtype=torch.int64)

        trimmed = trim_validation_tokens_for_smoke(tokens, seq_len=1024, val_max_tokens=0)

        self.assertTrue(torch.equal(trimmed, tokens))

    def test_trim_validation_tokens_for_smoke_caps_to_requested_full_sequences(self):
        tokens = torch.arange(4097, dtype=torch.int64)

        trimmed = trim_validation_tokens_for_smoke(tokens, seq_len=1024, val_max_tokens=2048)

        self.assertEqual(trimmed.numel(), 2049)
        self.assertTrue(torch.equal(trimmed, tokens[:2049]))

    def test_trim_validation_tokens_for_smoke_preserves_one_sequence_plus_target(self):
        tokens = torch.arange(4097, dtype=torch.int64)

        trimmed = trim_validation_tokens_for_smoke(tokens, seq_len=1024, val_max_tokens=1)

        self.assertEqual(trimmed.numel(), 1025)
        self.assertTrue(torch.equal(trimmed, tokens[:1025]))


if __name__ == "__main__":
    unittest.main()
