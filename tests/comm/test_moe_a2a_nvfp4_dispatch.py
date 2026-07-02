"""Correctness coverage for fused BF16-to-NVFP4 MoE dispatch."""

import pytest
import torch

from flashinfer.comm.trtllm_moe_alltoall import (
    get_moe_alltoall_module,
    moe_a2a_dispatch,
    moe_a2a_dispatch_nvfp4,
    moe_a2a_get_workspace_size_per_rank,
    moe_a2a_initialize,
)
from flashinfer.fp4_quantization import nvfp4_quantize
from flashinfer.tllm_enums import SfLayout
from flashinfer.utils import get_compute_capability


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)

TOP_K = 22
NUM_EXPERTS = 512
HIDDEN_SIZE = 8192


def _require_sm100() -> None:
    if get_compute_capability(torch.device("cuda"))[0] < 10:
        pytest.skip("fused NVFP4 dispatch requires SM100+")


def _make_routes(world_size: int, num_tokens: int) -> torch.Tensor:
    experts_per_rank = NUM_EXPERTS // world_size
    routes = []
    for rank in range(world_size):
        for token in range(num_tokens):
            destination_count = (1, 2, world_size)[token % 3]
            destinations = [
                (rank + token + offset) % world_size
                for offset in range(destination_count)
            ]
            routes.append(
                [
                    destinations[k % destination_count] * experts_per_rank
                    + ((token * TOP_K + k) % experts_per_rank)
                    for k in range(TOP_K)
                ]
            )
    return torch.tensor(routes, dtype=torch.int32, device="cuda")


def _allocate_workspace(world_size: int, max_num_tokens: int, extra_bytes: int = 0):
    dispatch_bytes = (
        HIDDEN_SIZE // 2 + HIDDEN_SIZE // 16 + TOP_K * 4 + TOP_K * 4 + extra_bytes
    )
    workspace_size = moe_a2a_get_workspace_size_per_rank(
        world_size,
        max_num_tokens,
        dispatch_bytes,
        HIDDEN_SIZE * 2,
    )
    workspace = torch.empty(
        world_size, workspace_size, dtype=torch.uint8, device="cuda"
    )
    metainfo = [
        moe_a2a_initialize(workspace, rank, world_size, max_num_tokens)
        for rank in range(world_size)
    ]
    return workspace, metainfo


def _launch_all_ranks(
    fused: bool,
    hidden_states: torch.Tensor,
    global_scale: torch.Tensor,
    routes: torch.Tensor,
    weights: torch.Tensor,
    token_ids: torch.Tensor,
    quantized: list[tuple[torch.Tensor, torch.Tensor]],
    workspace: torch.Tensor,
    metainfo: list[torch.Tensor],
    world_size: int,
    num_tokens: int,
):
    outputs = []
    streams = [torch.cuda.Stream() for _ in range(world_size)]
    for rank, stream in enumerate(streams):
        stream.wait_stream(torch.cuda.current_stream())
        token_slice = slice(rank * num_tokens, (rank + 1) * num_tokens)
        with torch.cuda.stream(stream):
            if fused:
                output, _ = moe_a2a_dispatch_nvfp4(
                    hidden_states[token_slice],
                    global_scale,
                    routes[token_slice],
                    [routes[token_slice], weights[token_slice], token_ids[token_slice]],
                    workspace,
                    metainfo[rank],
                    num_tokens,
                    rank,
                    world_size,
                    TOP_K,
                    NUM_EXPERTS,
                )
            else:
                packed, scales = quantized[rank]
                output, _ = moe_a2a_dispatch(
                    routes[token_slice],
                    [
                        packed,
                        scales,
                        routes[token_slice],
                        weights[token_slice],
                        token_ids[token_slice],
                    ],
                    workspace,
                    metainfo[rank],
                    num_tokens,
                    rank,
                    world_size,
                    TOP_K,
                    NUM_EXPERTS,
                )
            outputs.append(output)
    for stream in streams:
        stream.synchronize()
    return outputs


def _metainfo_indices() -> dict[str, int]:
    names, values = get_moe_alltoall_module().moe_a2a_get_metainfo_index_pairs()
    return {
        str(name).removeprefix("MOE_A2A_"): int(value)
        for name, value in zip(names, values, strict=True)
    }


def _recv_counts(
    workspace: torch.Tensor,
    metainfo: list[torch.Tensor],
    target_rank: int,
    world_size: int,
    indices: dict[str, int],
) -> torch.Tensor:
    offset = int(metainfo[target_rank][indices["RECV_COUNTERS_OFFSET_INDEX"]].item())
    return (
        workspace[target_rank, offset : offset + world_size * 4].view(torch.int32).cpu()
    )


def test_fused_nvfp4_dispatch_matches_materialized_dispatch(monkeypatch):
    _require_sm100()
    monkeypatch.delenv("FLASHINFER_NVFP4_4OVER6", raising=False)
    torch.manual_seed(20260702)
    world_size = 4
    num_tokens = 8
    total_tokens = world_size * num_tokens

    hidden_states = torch.randn(
        total_tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
    )
    global_scale = torch.tensor(
        [448.0 * 6.0 / hidden_states.abs().float().max().item()],
        dtype=torch.float32,
        device="cuda",
    )
    routes = _make_routes(world_size, num_tokens)
    token_ids = torch.arange(total_tokens, dtype=torch.int32, device="cuda")[:, None]
    weights = token_ids.to(torch.float32).expand(-1, TOP_K).contiguous()
    quantized = [
        nvfp4_quantize(
            hidden_states[rank * num_tokens : (rank + 1) * num_tokens],
            global_scale,
            sfLayout=SfLayout.layout_linear,
            enable_pdl=False,
        )
        for rank in range(world_size)
    ]

    baseline_workspace, baseline_metainfo = _allocate_workspace(
        world_size, num_tokens, token_ids.element_size()
    )
    fused_workspace, fused_metainfo = _allocate_workspace(
        world_size, num_tokens, token_ids.element_size()
    )
    baseline_outputs = _launch_all_ranks(
        False,
        hidden_states,
        global_scale,
        routes,
        weights,
        token_ids,
        quantized,
        baseline_workspace,
        baseline_metainfo,
        world_size,
        num_tokens,
    )
    fused_outputs = _launch_all_ranks(
        True,
        hidden_states,
        global_scale,
        routes,
        weights,
        token_ids,
        quantized,
        fused_workspace,
        fused_metainfo,
        world_size,
        num_tokens,
    )

    indices = _metainfo_indices()
    for target_rank in range(world_size):
        baseline_counts = _recv_counts(
            baseline_workspace,
            baseline_metainfo,
            target_rank,
            world_size,
            indices,
        )
        fused_counts = _recv_counts(
            fused_workspace,
            fused_metainfo,
            target_rank,
            world_size,
            indices,
        )
        torch.testing.assert_close(fused_counts, baseline_counts, atol=0, rtol=0)
        for source_rank, valid in enumerate(baseline_counts.tolist()):
            baseline_ids = baseline_outputs[target_rank][4][source_rank, :valid, 0]
            fused_ids = fused_outputs[target_rank][4][source_rank, :valid, 0]
            baseline_order = torch.argsort(baseline_ids)
            fused_order = torch.argsort(fused_ids)
            for payload_idx in range(5):
                torch.testing.assert_close(
                    fused_outputs[target_rank][payload_idx][source_rank, :valid][
                        fused_order
                    ],
                    baseline_outputs[target_rank][payload_idx][source_rank, :valid][
                        baseline_order
                    ],
                    atol=0,
                    rtol=0,
                )

        target_offset = int(
            baseline_metainfo[target_rank][
                indices["TOPK_TARGET_RANKS_OFFSET_INDEX"]
            ].item()
        )
        target_bytes = num_tokens * TOP_K * 4
        torch.testing.assert_close(
            fused_workspace[target_rank, target_offset : target_offset + target_bytes],
            baseline_workspace[
                target_rank, target_offset : target_offset + target_bytes
            ],
            atol=0,
            rtol=0,
        )


def test_fused_nvfp4_dispatch_cuda_graph_replay(monkeypatch):
    _require_sm100()
    monkeypatch.delenv("FLASHINFER_NVFP4_4OVER6", raising=False)
    torch.manual_seed(20260703)
    num_tokens = 4
    hidden_states = torch.randn(
        num_tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
    )
    base_hidden_states = hidden_states.clone()
    global_scale = torch.tensor([128.0], dtype=torch.float32, device="cuda")
    routes = _make_routes(1, num_tokens)
    token_ids = torch.arange(num_tokens, dtype=torch.int32, device="cuda")[:, None]
    workspace, metainfo = _allocate_workspace(1, num_tokens, token_ids.element_size())

    for _ in range(3):
        moe_a2a_dispatch_nvfp4(
            hidden_states,
            global_scale,
            routes,
            [routes, token_ids],
            workspace,
            metainfo[0],
            num_tokens,
            0,
            1,
            TOP_K,
            NUM_EXPERTS,
        )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, _ = moe_a2a_dispatch_nvfp4(
            hidden_states,
            global_scale,
            routes,
            [routes, token_ids],
            workspace,
            metainfo[0],
            num_tokens,
            0,
            1,
            TOP_K,
            NUM_EXPERTS,
        )

    for replay in range(16):
        hidden_states.copy_(base_hidden_states + replay / 64.0)
        graph.replay()
    torch.cuda.synchronize()

    expected_packed, expected_scales = nvfp4_quantize(
        hidden_states,
        global_scale,
        sfLayout=SfLayout.layout_linear,
        enable_pdl=False,
    )
    order = torch.argsort(output[3][0, :num_tokens, 0])
    source_order = output[3][0, :num_tokens, 0][order].to(torch.long)
    torch.testing.assert_close(
        output[0][0, :num_tokens][order],
        expected_packed[source_order],
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        output[1][0, :num_tokens][order],
        expected_scales[source_order],
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        output[2][0, :num_tokens][order], routes[source_order], atol=0, rtol=0
    )
