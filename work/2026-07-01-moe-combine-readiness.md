# MoE Combine Readiness Split

Date: 2026-07-01

Repo: FlashInfer v0.6.11 direct-output stack

Branch / base: `codex/fi-v0611-readiness` / `647ad0176825b014a0117c01010ca5e31f2203b3`

Hardware: Not available in this worktree environment

CUDA graph: Required validation mode

## Hypothesis

The combine data grid repeats peer-generation polling, a system acquire fence, and a CTA barrier
for every local token block. Moving readiness to one CTA per rank should reduce fixed work in the
data grid while retaining the same generation protocol and delayed-rank ordering.

## Changes

- Prepare still increments the local `uint32_t` generation and optionally stages payload data.
- A same-stream 64-thread readiness CTA publishes one generation flag per EP peer, waits for all
  peer flags using equality and the existing timeout, then performs system acquire ordering.
- The default data-kernel specialization contains no readiness polling, fence, or CTA barrier.
- `TLLM_MOE_A2A_SPLIT_COMBINE_READINESS=0` restores the legacy in-kernel protocol for A/B.

Implementation: [moeAlltoAllKernels.cu](../csrc/nv_internal/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.cu)

## Correctness Contract

1. All ranks launch readiness, including ranks with zero local tokens and workspace-backed payloads
   that require no prepare copy.
2. Same-stream ordering is `prepare -> readiness -> data`; no cooperative launch is used.
3. Readiness publishes only after a system release fence, polls with system-scope relaxed loads,
   and executes a system acquire fence after observing the matching peer generation.
4. Generation remains device-resident and increments modulo `2^32`, so CUDA graph replay and wrap
   behavior remain unchanged.
5. A missing or delayed rank blocks only the readiness CTA and retains the existing timeout/trap.
6. The combine output, dtype support, routing metadata, and public Python API are unchanged.

## Validation

Local execution was unavailable: this macOS host has no CUDA toolkit, GPU, PyTorch, or pytest.
Run the existing focused suite in the target image:

```bash
pytest -q tests/comm/test_trtllm_moe_alltoall.py -k 'combine or single_gpu'
```

The controlled multi-rank harness should additionally check:

- EP4 with local token vectors including `[0, N, N, N]` and all-zero participation.
- Delay each rank in turn before combine; verify no data kernel reads before the delayed publish.
- Seed generation near `UINT32_MAX`, cross wrap, and verify equality on each replay.
- Capture once and replay at least 100 times with static shapes and changing payload values.
- Compare default split mode against `TLLM_MOE_A2A_SPLIT_COMBINE_READINESS=0` bitwise where the
  accumulation order is identical.

## Performance

No performance claim is made without same-system measurements.

| Case | Baseline | Candidate | Delta | Notes |
|---|---:|---:|---:|---|
| Ultra NVFP4, H=8192, top-k=22, 512 experts, EP4 | pending | pending | pending | CUDA graph required |

Benchmark both split values in separate processes because environment values are cached. Sweep
local tokens `{0, 1, 2, 4, 8, 16, 32, 64, 128}`, eager versus graph replay,
`TLLM_MOE_A2A_ONE_BLOCK_PER_TOKEN={0,1}`, and
`TLLM_MOE_A2A_COMBINE_BLOCK_SIZE={128,256,512}`. Record combine end-to-end latency and the
readiness/data kernel durations separately; report median and p90 after warmup.

## Risks

- The extra launch can lose at very small token counts even if the data kernel becomes cheaper.
- Runtime fallback specialization doubles the combine kernel variant count and can increase JIT
  compile time and cache size.
- Correctness depends on all EP ranks entering each generation; this is unchanged from baseline.

## Sources

- [PTX ISA memory consistency: release and acquire patterns](https://docs.nvidia.com/cuda/parallel-thread-execution/#release-and-acquire-patterns)
- [CUDA Programming Guide: stream ordering and system-scope fences](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
