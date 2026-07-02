# EP4 Top-k22 Combine Specialization

Date: 2026-07-02
Repo: `/Users/snordmann/repos/experiments/fi-opt12-07-combine-ep4`
Branch: `snordmann/fi-opt12-07-combine-ep4-20260702`
Base: `04bf64bd`
Hardware target: one GB200 node, TP1/DP4/EP4
CUDA graph: required for correctness and timing validation

## Hypothesis

For BF16, unquantized EP4/top-k22 combine, replacing the generic 64-rank pointer table with a
compact four-rank kernel argument and making EP size a template constant can reduce parameter-table
addressing and readiness-loop overhead. The benefit is unmeasured and may be offset by the four
cached receive pointers increasing register pressure.

## Changes

- Route only `ep_size=4`, `top_k=22`, BF16, unquantized, linear-layout combine to the specialization.
- Cache the four receive-buffer pointers and select among them without dynamically indexing the
  generic `kMaxRanks x kMaxPayloads` table.
- Keep the sparse 22 metadata slots and existing reduction tree unchanged.
- Retain the original kernel for every other parameter tuple.

Source: [moeAlltoAllKernels.cu](../csrc/nv_internal/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.cu)

## Correctness

Static checks pass. GPU compilation, EP4 correctness, skewed/zero-token behavior, and CUDA graph
capture/replay have not been run in this worktree.

Target-path test definition:
[test_trtllm_moe_alltoall.py](../tests/comm/test_trtllm_moe_alltoall.py)

## Performance

| Case | Baseline | Candidate | Delta | Notes |
|---|---:|---:|---:|---|
| EP4/top-k22/BF16/H8192 | Not run | Not run | Not available | Do not claim performance |

## Exact Microbenchmark Signature

```text
API: flashinfer.comm.moe_a2a_combine(..., output=caller_output)
GPU topology: one GB200 node, four rank-pinned GPUs, TP1/DP4/EP4
ep_size: 4
ep_rank: 0..3
num_experts: 512 contiguous, 128 per rank
top_k: 22, sparse first-rank-occurrence metadata retained
payload dtype / output dtype: torch.bfloat16 / torch.bfloat16
elements_per_token: 8192
runtime_max_tokens_per_rank: 32768
local_num_tokens matrix: (4,4,4,4), (18937,32768,32768,32768),
                         (32768,32768,32768,32768)
payload_in_workspace: false
output_dtype: None
output_scales: None
output_scalar_scale: 1.0
sf_layout: SfLayout.layout_linear
TLLM_MOE_A2A_COMBINE_BLOCK_SIZE: 256
execution: CUDA graph capture and replay; alternate baseline/candidate order in one process;
           report maximum-rank CUDA-event latency per replay
timing: 20 warmup blocks, 80 paired measured blocks, 100 graph replays per block
correctness: exact BF16 output against baseline for eager calls and 1,000 graph replays,
             including one zero-token rank and asymmetric rank arrival
```

The primary metric is paired median graph-replay latency for the public combine call. Also record
the combine kernel's register count and post-last-rank interval with Nsight Systems; a register
increase or an unchanged critical interval can invalidate the hypothesis even if raw medians move.

## Risks

- Four cached 64-bit pointers can increase registers or spills.
- The fixed-pointer selector assumes valid EP4 routing metadata; dispatch already guarantees ranks
  in `[0, 4)` and marks duplicate rank slots with `dst_idx=-1`.
- Static checks do not establish CUDA compilation or graph safety.
- Server-level throughput is noisy; a microbenchmark result alone is insufficient for an e2e claim.

## Sources

- [Current CUDA implementation](../csrc/nv_internal/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.cu)
- [Focused target test](../tests/comm/test_trtllm_moe_alltoall.py)
- [Prior one-sided optimization report](/Users/snordmann/Documents/DL%20Compilers/reports/2026-07-01-flashinfer-one-sided-optimization-experiments.md)
- [Routing decision follow-up](/Users/snordmann/Documents/DL%20Compilers/reports/2026-07-02-flashinfer-routing-decision-followup.md)
