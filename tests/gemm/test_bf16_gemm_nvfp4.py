import pytest
import torch


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


@pytest.fixture(autouse=True)
def require_supported_compute_capability():
    from flashinfer.gemm import bf16_gemm_nvfp4

    major, minor = torch.cuda.get_device_capability()
    capability = major * 10 + minor
    if not bf16_gemm_nvfp4.is_compute_capability_supported(capability):
        pytest.skip(f"bf16_gemm_nvfp4 does not support compute capability {capability}")


def _reference(input, weight, global_scale):
    from flashinfer.quantization.fp4_quantization import fp4_quantize

    projected = torch.nn.functional.linear(input, weight)
    return fp4_quantize(
        projected,
        global_scale,
        is_sf_swizzled_layout=False,
        enable_pdl=False,
    )


def _dequantize_linear_nvfp4(packed, scales, global_scale):
    codebook = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] * 2,
        device=packed.device,
        dtype=torch.float32,
    )
    codebook[8:] *= -1
    codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).reshape(
        packed.shape[0], -1
    )
    values = codebook[codes.long()]
    block_scales = scales.float().repeat_interleave(16, dim=-1)
    return values * block_scales / global_scale.float()


def _assert_matches_reference(actual, expected, projected, global_scale):
    actual_q, actual_sf = actual
    expected_q, expected_sf = expected
    scale_match = (
        (actual_sf.view(torch.uint8) == expected_sf.view(torch.uint8)).float().mean()
    )
    quant_match = (actual_q == expected_q).float().mean()
    assert scale_match >= 0.99
    # MMA accumulation order can move values across an adjacent FP4 threshold,
    # so bitwise identity is not required. Current SM100 evidence is >=99.51%.
    assert quant_match >= 0.99

    dequantized = _dequantize_linear_nvfp4(actual_q, actual_sf, global_scale)
    relative_l2 = torch.linalg.vector_norm(dequantized - projected.float())
    relative_l2 /= torch.linalg.vector_norm(projected.float())
    assert relative_l2 < 0.2


def test_bf16_gemm_nvfp4_advertises_only_validated_architecture():
    from flashinfer.gemm import bf16_gemm_nvfp4

    assert bf16_gemm_nvfp4.is_compute_capability_supported(100)
    assert not bf16_gemm_nvfp4.is_compute_capability_supported(90)
    assert not bf16_gemm_nvfp4.is_compute_capability_supported(103)


@pytest.mark.parametrize("num_tokens", [1, 4, 17, 64])
def test_bf16_gemm_nvfp4_matches_unfused(num_tokens):
    from flashinfer.gemm import bf16_gemm_nvfp4

    torch.manual_seed(7)
    input = torch.randn(num_tokens, 8192, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(2048, 8192, device="cuda", dtype=torch.bfloat16) / 64
    global_scale = torch.tensor([448.0], device="cuda", dtype=torch.float32)

    actual = bf16_gemm_nvfp4(input, weight, global_scale)
    projected = torch.nn.functional.linear(input, weight)
    expected = _reference(input, weight, global_scale)
    _assert_matches_reference(actual, expected, projected, global_scale)


@pytest.mark.parametrize("num_tokens", [1, 4, 17, 64])
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
    projected = torch.nn.functional.linear(input, weight)
    _assert_matches_reference(
        first,
        _reference(input, weight, global_scale),
        projected,
        global_scale,
    )

    input.normal_()
    graph.replay()
    second = (output.clone(), output_scale.clone())
    assert not torch.equal(first[0], second[0])
    projected = torch.nn.functional.linear(input, weight)
    _assert_matches_reference(
        second,
        _reference(input, weight, global_scale),
        projected,
        global_scale,
    )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two GPUs are required")
@pytest.mark.parametrize("cross_device_buffer", ["output", "output_scale"])
def test_bf16_gemm_nvfp4_rejects_cross_device_output(cross_device_buffer):
    from flashinfer.gemm import bf16_gemm_nvfp4

    input = torch.empty(1, 8192, device="cuda:0", dtype=torch.bfloat16)
    weight = torch.empty(2048, 8192, device="cuda:0", dtype=torch.bfloat16)
    global_scale = torch.ones(1, device="cuda:0", dtype=torch.float32)
    output = torch.empty(1, 1024, device="cuda:0", dtype=torch.uint8)
    output_scale = torch.empty(1, 128, device="cuda:0", dtype=torch.float8_e4m3fn)

    if cross_device_buffer == "output":
        output = output.to("cuda:1")
    else:
        output_scale = output_scale.to("cuda:1")

    with pytest.raises(ValueError, match="must be on the input device"):
        bf16_gemm_nvfp4(
            input,
            weight,
            global_scale,
            output=output,
            output_scale=output_scale,
        )
