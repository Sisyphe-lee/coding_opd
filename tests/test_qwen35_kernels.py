from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from coding_opd.qwen35_kernels import (
    _install_device_flops_override,
    _refresh_qwen35_training_fast_path,
    _seq_idx_to_cu_seqlens,
)


def test_seq_idx_to_cu_seqlens() -> None:
    seq_idx = torch.tensor([[0, 0, 1, 1, 1, 4]], dtype=torch.int32)
    assert _seq_idx_to_cu_seqlens(seq_idx).tolist() == [0, 2, 5, 6]


def test_seq_idx_to_cu_seqlens_rejects_batched_ids() -> None:
    with pytest.raises(ValueError, match="shape"):
        _seq_idx_to_cu_seqlens(torch.zeros((2, 3), dtype=torch.int32))


def test_device_flops_override_uses_explicit_peak(monkeypatch: pytest.MonkeyPatch) -> None:
    from verl.utils import flops_counter

    monkeypatch.setenv("CODING_OPD_DEVICE_PEAK_TFLOPS", "2250")
    monkeypatch.setattr(flops_counter, "get_device_flops", lambda *args, **kwargs: -1.0)

    _install_device_flops_override()

    assert flops_counter.get_device_flops("T") == 2250.0
    assert flops_counter.get_device_flops("P") == 2.25


def test_refresh_qwen35_training_fast_path_replaces_stale_import_gate() -> None:
    modeling = SimpleNamespace(
        causal_conv1d_fn=object(),
        chunk_gated_delta_rule=object(),
        fused_recurrent_gated_delta_rule=object(),
        is_fast_path_available=False,
    )

    _refresh_qwen35_training_fast_path(modeling)

    assert modeling.is_fast_path_available
