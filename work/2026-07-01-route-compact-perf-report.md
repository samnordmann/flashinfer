# Compact MoE Route Perf Report

Date: 2026-07-01

Repo: FlashInfer v0.6.11 direct-output stack

Branch: `codex/fi-v0611-route-compact`

Base commit: `647ad0176825b014a0117c01010ca5e31f2203b3`

Hardware: Not run locally; parent owns the controlled GPU harness

CUDA graph: Intended for capture/replay; no allocation, host synchronization, or launch-shape
mutation was added

## Hypothesis

For Ultra NVFP4 with `H=8192`, `top_k=22`, and EP4, the direct-output A2A kernels spend
unnecessary instructions and registers constructing and consuming 22 route positions even though
there can be at most four physical destinations. Building a dense route once in the producer
thread and specializing consumers to four destinations should reduce route work, shared memory,
fanout pointer state, and combine accumulator state.

## Changes

- One producer thread per token deduplicates target ranks and reserves destination slots.
- Routes store first-seen unique ranks densely, followed by `-1` padding through the active
  destination capacity.
- EP4 dispatch and combine use `min(TOP_K, 4)` compile-time destination capacity.
- Other EP sizes retain the generic `TOP_K` capacity fallback.

## Correctness

Focused command for the target topology:

```bash
mpirun -np 4 pytest tests/comm/test_mnnvl_moe_alltoall.py -v -s
```

The test now validates dense first-seen rank ordering and includes `top_k=22` dispatch/combine
coverage. It requires MNNVL-capable hardware and was not run in this local environment.

## Performance

No measurements are claimed. Run baseline and candidate in the same parent-controlled harness
with `num_experts=512`, `top_k=22`, `ep_size=4`, `hidden_size=8192`, Ultra NVFP4, direct output,
and CUDA graphs enabled. Sweep token count/batch, dispatch/combine block-vs-warp policy, warmup,
replay count, and payload-in-workspace mode. Record end-to-end TPOT plus dispatch and combine kernel
latency; inspect register count, occupancy, shared memory, and local-memory spills.

## Risks

- Compacting routes changes floating-point combine order relative to sparse top-k positions, so
  BF16/FP16 results can differ within normal reduction tolerance.
- Specialization increases generated kernel variants for EP4.
- Performance is unverified until the controlled GB200/GV200 harness runs.

## Sources

- [MoE A2A kernels](../csrc/nv_internal/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.cu)
- [MoE A2A kernel contract](../csrc/nv_internal/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.h)
- [MNNVL MoE A2A correctness tests](../tests/comm/test_mnnvl_moe_alltoall.py)
