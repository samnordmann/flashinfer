"""Distributed correctness check for direct R128C4 MoE scale dispatch.

Run on one four-GPU NVLink node with::

    torchrun --standalone --nproc-per-node=4 \
      tests/comm/run_trtllm_moe_alltoall_r128c4_correctness.py

The check reconstructs the exact logical dispatch output from the routing metadata,
applies the existing ``block_scale_interleave`` implementation, and compares every
physical output byte with direct R128C4 dispatch. It covers eager execution and CUDA
graph replay at row-padding boundaries for the Nemotron Ultra EP4 shape.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist

from flashinfer.comm.mapping import Mapping
from flashinfer.comm.mnnvl import MnnvlConfig, TorchDistBackend
from flashinfer.comm.trtllm_moe_alltoall import (
    MoeA2APayloadLayout,
    MoeAlltoAll,
    moe_a2a_dispatch,
    moe_a2a_get_dispatch_payload_size,
    moe_a2a_get_workspace_size_per_rank,
)
from flashinfer.jit import env as jit_env
from flashinfer.quantization.fp4_quantization import block_scale_interleave


EP_SIZE = 4
NUM_EXPERTS = 512
TOP_K = 22
SCALE_COLUMNS = 8192 // 16


@dataclass
class DispatchResult:
    physical: torch.Tensor
    target_ranks: torch.Tensor
    send_indices: torch.Tensor


def _make_routes(num_tokens: int, rank: int) -> torch.Tensor:
    token = torch.arange(num_tokens, dtype=torch.int32).unsqueeze(1)
    choice = torch.arange(TOP_K, dtype=torch.int32).unsqueeze(0)
    target_rank = choice.remainder(EP_SIZE)
    local_expert = (token * TOP_K + choice + rank).remainder(NUM_EXPERTS // EP_SIZE)
    return (target_rank * (NUM_EXPERTS // EP_SIZE) + local_expert).cuda()


def _make_payload(num_tokens: int, rank: int, generation: int = 0) -> torch.Tensor:
    token = torch.arange(num_tokens, dtype=torch.int32).unsqueeze(1)
    column = torch.arange(SCALE_COLUMNS, dtype=torch.int32).unsqueeze(0)
    values = (rank * 37 + token * 11 + column * 3 + generation * 17).remainder(251)
    return values.to(torch.uint8).cuda()


def _workspace_i32_view(
    moe_a2a: MoeAlltoAll, name: str, num_tokens: int
) -> torch.Tensor:
    assert moe_a2a._METAINFO_INDEX is not None
    offset_index = moe_a2a._METAINFO_INDEX[name]
    byte_offset = int(moe_a2a.metainfo[offset_index])
    byte_count = num_tokens * TOP_K * torch.int32.itemsize
    return (
        moe_a2a.workspace[moe_a2a.ep_rank, byte_offset : byte_offset + byte_count]
        .view(torch.int32)
        .view(num_tokens, TOP_K)
    )


def _dispatch_direct(
    moe_a2a: MoeAlltoAll,
    routes: torch.Tensor,
    payload: torch.Tensor,
    num_tokens: int,
) -> DispatchResult:
    (physical,), _ = moe_a2a_dispatch(
        routes,
        [payload],
        moe_a2a.workspace,
        moe_a2a.metainfo,
        num_tokens,
        moe_a2a.ep_rank,
        EP_SIZE,
        TOP_K,
        NUM_EXPERTS,
        [MoeA2APayloadLayout.R128C4],
    )
    torch.cuda.synchronize()
    return DispatchResult(
        physical=physical.clone(),
        target_ranks=_workspace_i32_view(
            moe_a2a, "TOPK_TARGET_RANKS_OFFSET_INDEX", num_tokens
        ).cpu(),
        send_indices=_workspace_i32_view(
            moe_a2a, "TOPK_SEND_INDICES_OFFSET_INDEX", num_tokens
        ).cpu(),
    )


def _reference_from_routing(
    result: DispatchResult,
    payload: torch.Tensor,
    num_tokens: int,
    rank: int,
) -> torch.Tensor:
    local_state = (payload.cpu(), result.target_ranks, result.send_indices)
    all_state: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None] = [
        None
    ] * EP_SIZE
    dist.all_gather_object(all_state, local_state)

    linear = torch.zeros(
        (EP_SIZE * num_tokens, SCALE_COLUMNS), dtype=torch.uint8, device="cuda"
    )
    for source_rank, state in enumerate(all_state):
        assert state is not None
        source_payload, target_ranks, send_indices = state
        source_payload = source_payload.cuda()
        for token in range(num_tokens):
            for choice in range(TOP_K):
                if int(target_ranks[token, choice]) != rank:
                    continue
                destination = int(send_indices[token, choice])
                assert 0 <= destination < num_tokens
                linear[source_rank * num_tokens + destination].copy_(
                    source_payload[token]
                )
    return block_scale_interleave(linear)


def _check_result(
    result: DispatchResult,
    payload: torch.Tensor,
    num_tokens: int,
    rank: int,
    phase: str,
) -> None:
    expected = _reference_from_routing(result, payload, num_tokens, rank)
    torch.testing.assert_close(result.physical, expected, rtol=0, atol=0)
    if rank == 0:
        print(
            f"PASS {phase}: tokens_per_rank={num_tokens}, "
            f"physical_bytes={result.physical.numel()}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--token-counts",
        type=int,
        nargs="+",
        default=[1, 31, 32, 33, 63, 64, 65],
        help="Per-rank token counts around 128-row EP4 padding boundaries.",
    )
    parser.add_argument("--graph-replays", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != EP_SIZE:
        raise RuntimeError(f"expected {EP_SIZE} ranks, got {world_size}")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    aot_dir = jit_env.FLASHINFER_AOT_DIR
    if os.environ.get("FLASHINFER_FORCE_JIT_MOE_A2A") == "1":
        # Force only the MoE A2A module to use the mounted candidate sources.
        # Restore the AOT path before loading the reference interleave kernel.
        jit_env.FLASHINFER_AOT_DIR = Path("/nonexistent/flashinfer-aot")
    max_tokens = max(args.token_counts)
    max_payload_size = moe_a2a_get_dispatch_payload_size(
        EP_SIZE,
        max_tokens,
        SCALE_COLUMNS,
        torch.uint8.itemsize,
        MoeA2APayloadLayout.R128C4,
    )
    workspace_size = moe_a2a_get_workspace_size_per_rank(
        EP_SIZE,
        max_tokens,
        SCALE_COLUMNS,
        0,
        dispatch_payload_sizes=[max_payload_size],
    )
    mapping = Mapping(
        world_size=EP_SIZE,
        rank=rank,
        tp_size=EP_SIZE,
        moe_tp_size=1,
        moe_ep_size=EP_SIZE,
    )
    moe_a2a = MoeAlltoAll(
        mapping,
        max_tokens,
        TOP_K,
        NUM_EXPERTS,
        workspace_size_per_rank=workspace_size,
        mnnvl_config=MnnvlConfig(comm_backend=TorchDistBackend()),
    )
    jit_env.FLASHINFER_AOT_DIR = aot_dir

    for num_tokens in sorted(args.token_counts):
        routes = _make_routes(num_tokens, rank)
        payload = _make_payload(num_tokens, rank)
        dist.barrier()
        result = _dispatch_direct(moe_a2a, routes, payload, num_tokens)
        _check_result(result, payload, num_tokens, rank, "eager")

    graph_tokens = max_tokens
    routes = _make_routes(graph_tokens, rank)
    payload = _make_payload(graph_tokens, rank)
    dist.barrier()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        (graph_output,), _ = moe_a2a_dispatch(
            routes,
            [payload],
            moe_a2a.workspace,
            moe_a2a.metainfo,
            graph_tokens,
            rank,
            EP_SIZE,
            TOP_K,
            NUM_EXPERTS,
            [MoeA2APayloadLayout.R128C4],
        )

    for replay in range(args.graph_replays):
        payload.copy_(_make_payload(graph_tokens, rank, replay + 1))
        dist.barrier()
        graph.replay()
        torch.cuda.synchronize()
        result = DispatchResult(
            physical=graph_output.clone(),
            target_ranks=_workspace_i32_view(
                moe_a2a, "TOPK_TARGET_RANKS_OFFSET_INDEX", graph_tokens
            ).cpu(),
            send_indices=_workspace_i32_view(
                moe_a2a, "TOPK_SEND_INDICES_OFFSET_INDEX", graph_tokens
            ).cpu(),
        )
        _check_result(result, payload, graph_tokens, rank, f"graph[{replay}]")

    dist.barrier()
    if rank == 0:
        print("All direct R128C4 dispatch checks passed")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
