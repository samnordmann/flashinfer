"""Target-specialized BF16 projection with a linear NVFP4 epilogue."""

import functools
from types import SimpleNamespace
from typing import Optional

import torch

from ..api_logging import flashinfer_api
from ..jit.gemm import gen_bf16_gemm_nvfp4_module
from ..utils import register_custom_op


_HIDDEN_IN = 8192
_HIDDEN_OUT = 2048
_MAX_TOKENS = 256


def _check_inputs(
    input: torch.Tensor,
    weight: torch.Tensor,
    global_scale: torch.Tensor,
) -> None:
    if input.dim() != 2 or input.shape[1] != _HIDDEN_IN:
        raise ValueError(f"input must have shape (M, {_HIDDEN_IN})")
    if input.shape[0] < 1 or input.shape[0] > _MAX_TOKENS:
        raise ValueError(f"input rows must be in [1, {_MAX_TOKENS}]")
    if weight.shape != (_HIDDEN_OUT, _HIDDEN_IN):
        raise ValueError(f"weight must have shape ({_HIDDEN_OUT}, {_HIDDEN_IN})")
    if input.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("input and weight must be bfloat16")
    if not input.is_contiguous() or not weight.is_contiguous():
        raise ValueError("input and weight must be contiguous")
    if global_scale.dtype != torch.float32 or global_scale.numel() != 1:
        raise TypeError("global_scale must be a one-element float32 tensor")
    if input.device != weight.device or input.device != global_scale.device:
        raise ValueError("input, weight, and global_scale must be on the same device")


@functools.cache
def _get_module():
    module = gen_bf16_gemm_nvfp4_module().build_and_load()

    @register_custom_op(
        "flashinfer::bf16_gemm_nvfp4_op",
        mutates_args=["output", "output_scale"],
    )
    def op(
        input: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor,
        output_scale: torch.Tensor,
        global_scale: torch.Tensor,
        launch_with_pdl: bool,
    ) -> None:
        module.bf16_gemm_nvfp4(
            input,
            weight,
            output,
            output_scale,
            global_scale,
            launch_with_pdl,
        )

    return SimpleNamespace(op=op)


@flashinfer_api
def bf16_gemm_nvfp4(
    input: torch.Tensor,
    weight: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    output: Optional[torch.Tensor] = None,
    output_scale: Optional[torch.Tensor] = None,
    launch_with_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ``input @ weight.T`` and emit linear NVFP4 data and scales.

    The specialized shape is ``K=8192, N=2048, 1 <= M <= 256``. Accumulators
    are rounded to BF16 before block-16 NVFP4 quantization to match the
    unfused projection-output contract.
    """
    _check_inputs(input, weight, global_scale)
    rows = input.shape[0]
    if output is None:
        output = torch.empty(
            (rows, _HIDDEN_OUT // 2), dtype=torch.uint8, device=input.device
        )
    if output_scale is None:
        output_scale = torch.empty(
            (rows, _HIDDEN_OUT // 16),
            dtype=torch.float8_e4m3fn,
            device=input.device,
        )
    if output.shape != (rows, _HIDDEN_OUT // 2) or output.dtype != torch.uint8:
        raise ValueError("output must be contiguous uint8 with shape (M, 1024)")
    if (
        output_scale.shape != (rows, _HIDDEN_OUT // 16)
        or output_scale.dtype != torch.float8_e4m3fn
    ):
        raise ValueError(
            "output_scale must be contiguous float8_e4m3fn with shape (M, 128)"
        )
    if not output.is_contiguous() or not output_scale.is_contiguous():
        raise ValueError("output and output_scale must be contiguous")
    _get_module().op(
        input,
        weight,
        output,
        output_scale,
        global_scale,
        launch_with_pdl,
    )
    return output, output_scale


__all__ = ["bf16_gemm_nvfp4"]
