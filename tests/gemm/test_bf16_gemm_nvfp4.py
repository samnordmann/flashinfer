import pytest
import torch


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
    pytest.mark.parametrize("num_tokens", [1, 4, 17, 64]),
]


def _reference(input, weight, global_scale):
    from flashinfer.quantization.fp4_quantization import fp4_quantize

    projected = torch.nn.functional.linear(input, weight)
    return fp4_quantize(
        projected,
        global_scale,
        is_sf_swizzled_layout=False,
        enable_pdl=False,
    )


def _assert_matches_reference(actual, expected):
    actual_q, actual_sf = actual
    expected_q, expected_sf = expected
    scale_match = (
        (actual_sf.view(torch.uint8) == expected_sf.view(torch.uint8)).float().mean()
    )
    quant_match = (actual_q == expected_q).float().mean()
    assert scale_match >= 0.99
    assert quant_match >= 0.95


def test_bf16_gemm_nvfp4_matches_unfused(num_tokens):
    from flashinfer.gemm import bf16_gemm_nvfp4

    torch.manual_seed(7)
    input = torch.randn(num_tokens, 8192, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(2048, 8192, device="cuda", dtype=torch.bfloat16) / 64
    global_scale = torch.tensor([448.0], device="cuda", dtype=torch.float32)

    actual = bf16_gemm_nvfp4(input, weight, global_scale)
    expected = _reference(input, weight, global_scale)
    _assert_matches_reference(actual, expected)


def test_bf16_gemm_nvfp4_changing_input_cuda_graph(num_tokens):
    from flashinfer.gemm import bf16_gemm_nvfp4

    torch.manual_seed(11)
    input = torch.randn(num_tokens, 8192, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(2048, 8192, device="cuda", dtype=torch.bfloat16) / 64
    global_scale = torch.tensor([448.0], device="cuda", dtype=torch.float32)
    output = torch.empty(num_tokens, 1024, device="cuda", dtype=torch.uint8)
    output_scale = torch.empty(
        num_tokens, 128, device="cuda", dtype=torch.float8_e4m3fn
    )

    bf16_gemm_nvfp4(
        input,
        weight,
        global_scale,
        output=output,
        output_scale=output_scale,
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        bf16_gemm_nvfp4(
            input,
            weight,
            global_scale,
            output=output,
            output_scale=output_scale,
        )

    graph.replay()
    first = (output.clone(), output_scale.clone())
    _assert_matches_reference(first, _reference(input, weight, global_scale))

    input.normal_()
    graph.replay()
    second = (output.clone(), output_scale.clone())
    assert not torch.equal(first[0], second[0])
    _assert_matches_reference(second, _reference(input, weight, global_scale))
