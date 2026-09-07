"""Safe Qwen3.5 training-kernel integration for B300.

The external causal-conv1d CUDA extension has a known native-sm_103 correctness
problem. FLA's Triton convolution avoids that extension while also avoiding the
per-sequence PyTorch conv1d fallback used by Transformers/veRL.
"""

from __future__ import annotations

import os

import torch


def _seq_idx_to_cu_seqlens(seq_idx: torch.Tensor) -> torch.Tensor:
    """Convert causal-conv1d sequence IDs to FLA packed-sequence offsets."""
    if seq_idx.ndim != 2 or seq_idx.shape[0] != 1:
        raise ValueError(f"Expected seq_idx with shape [1, tokens], got {tuple(seq_idx.shape)}")
    token_count = seq_idx.shape[1]
    if token_count == 0:
        return torch.zeros(1, device=seq_idx.device, dtype=torch.int32)
    boundaries = torch.nonzero(seq_idx[0, 1:] != seq_idx[0, :-1], as_tuple=False).flatten() + 1
    endpoints = torch.tensor([0, token_count], device=seq_idx.device, dtype=boundaries.dtype)
    return torch.cat((endpoints[:1], boundaries, endpoints[1:])).to(torch.int32)


def fla_causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    seq_idx: torch.Tensor | None = None,
    **_: object,
) -> torch.Tensor:
    """Adapt Transformers' channel-first API to FLA's Triton convolution."""
    from fla.modules.conv import causal_conv1d

    cu_seqlens = _seq_idx_to_cu_seqlens(seq_idx) if seq_idx is not None else None
    output, _final_state = causal_conv1d(
        x=x.transpose(1, 2),
        weight=weight,
        bias=bias,
        activation=activation,
        backend="triton",
        cu_seqlens=cu_seqlens,
    )
    return output.transpose(1, 2)


def install_qwen35_fla_kernels() -> None:
    """Install FLA kernels before Transformers constructs Qwen3.5 modules."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    from transformers.models.qwen3_5 import modeling_qwen3_5

    modeling_qwen3_5.causal_conv1d_fn = fla_causal_conv1d_fn
    modeling_qwen3_5.chunk_gated_delta_rule = chunk_gated_delta_rule
    modeling_qwen3_5.fused_recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule
    _refresh_qwen35_training_fast_path(modeling_qwen3_5)

    _install_device_flops_override()


def _refresh_qwen35_training_fast_path(modeling_qwen3_5: object) -> None:
    """Refresh Transformers' import-time gate after installing training kernels.

    Transformers computes ``is_fast_path_available`` before this plugin replaces
    the missing native causal-conv function. The stale value only emits a false
    fallback warning, but that warning obscures which kernels the layer actually
    binds. Cached one-token decoding may still use Transformers' torch update;
    actor training uses the three multi-token functions checked here.
    """
    required = (
        getattr(modeling_qwen3_5, "causal_conv1d_fn", None),
        getattr(modeling_qwen3_5, "chunk_gated_delta_rule", None),
        getattr(modeling_qwen3_5, "fused_recurrent_gated_delta_rule", None),
    )
    setattr(modeling_qwen3_5, "is_fast_path_available", all(required))


def _install_device_flops_override() -> None:
    """Override veRL's peak-FLOPS lookup when the cluster masks the GPU name."""
    peak_tflops = os.environ.get("CODING_OPD_DEVICE_PEAK_TFLOPS")
    if peak_tflops is None:
        return

    peak_flops = float(peak_tflops) * 1e12
    unit_divisors = {
        "B": 1e9,
        "K": 1e3,
        "M": 1e6,
        "G": 1e9,
        "T": 1e12,
        "P": 1e15,
    }

    from verl.utils import flops_counter

    def get_device_flops(unit: str = "T", device_name: str | None = None) -> float:
        del device_name
        try:
            divisor = unit_divisors[unit]
        except KeyError as exc:
            raise ValueError(f"Unsupported FLOPS unit: {unit}") from exc
        return peak_flops / divisor

    flops_counter.get_device_flops = get_device_flops
