# Combine Readiness Poll Backoff Experiment

Date: 2026-07-02
Repo: `/Users/snordmann/repos/experiments/fi-opt12-12-independent-audit`
Branch: `snordmann/fi-opt12-12-independent-audit-20260702`
Base: `04bf64bd2b7f65effab7dc52831a628c1cbf5046`
Hardware target: one GB200 node, TP1/DP4/EP4
Model shape: Nemotron Ultra NVFP4, H8192, 512 experts, top-k 22
CUDA graph: required for correctness and performance runs

## Hypothesis

The baseline combine launches one CTA per local token. Until all EP ranks publish
the current generation, every resident CTA's first warp repeatedly loads the same
system-scope completion flags. Baseline profiles show milliseconds of rank-arrival
skew, so this can sustain unnecessary MNNVL coherence traffic.

After a failed poll, use CUDA's bounded exponential nanosleep sequence
`8, 16, 32, 64, 128, 256 ns`. A flag that is already ready takes the unchanged
single-load path. The maximum added detection delay after reaching the cap is
approximately 256 ns.

## Changes

- Back off only the combine readiness loop after a failed flag comparison.
- Keep dispatch polling, generation values, flag addresses, system fences,
  timeout handling, launch geometry, payloads, routes, and reduction order unchanged.
- Compile the sleep only for compute capability 7.0 or newer; the older-arch path
  remains the original tight poll.

## Correctness

Performance is unmeasured. Required correctness gate:

```bash
CANDIDATE_ROOT=/lustre/fsw/network_software_cloudai/snordmann/repos/flashinfer-opt12-combine-backoff \
TOKEN_COUNTS=0,7,13,31 DELAY_RANK=0 DELAY_RANK_MS=5 GRAPH_REPLAYS=1000 \
  sbatch --export=ALL \
  /lustre/fsw/network_software_cloudai/snordmann/repos/_agent_playbook/fi_cta_vllm_scripts/flashinfer_one_sided_candidate_correctness.sbatch
```

This must run through the existing source-JIT wrapper and GB200 MNNVL container.
It changes every producer payload before each graph replay, so stale remote reads
are observable. The patch adds no allocation, host read, mutable state, kernel,
or graph node and is therefore graph-capture neutral by construction.

Local validation: `git diff --check` and the repository commit hooks passed,
including `clang-format`. This workstation has no `nvcc` or GPU, so CUDA
compilation and runtime correctness remain unverified.

## Performance

| Case | Baseline | Candidate | Delta | Notes |
|---|---:|---:|---:|---|
| EP4 combine, 4/64/512/8192 tokens | unmeasured | unmeasured | unmeasured | paired AB and BA required |
| Nemotron Ultra vLLM throughput/TPOT | unmeasured | unmeasured | unmeasured | only run after microbenchmark gate |

Exact paired microbenchmark submission:

```bash
sbatch --export=ALL,\
BASELINE_REPO=/lustre/fsw/network_software_cloudai/snordmann/repos/flashinfer-opt12-baseline,\
CANDIDATE_REPO=/lustre/fsw/network_software_cloudai/snordmann/repos/flashinfer-opt12-combine-backoff,\
CANDIDATE_LABEL=combine-backoff,ORDER=AB,\
TOKEN_COUNTS=4:64:512:8192,BENCH_BLOCKS=80,WARMUP_BLOCKS=10 \
  /lustre/fsw/network_software_cloudai/snordmann/repos/_agent_playbook/flashinfer_opt12_20260702/paired_micro.sbatch

sbatch --export=ALL,\
BASELINE_REPO=/lustre/fsw/network_software_cloudai/snordmann/repos/flashinfer-opt12-baseline,\
CANDIDATE_REPO=/lustre/fsw/network_software_cloudai/snordmann/repos/flashinfer-opt12-combine-backoff,\
CANDIDATE_LABEL=combine-backoff,ORDER=BA,\
TOKEN_COUNTS=4:64:512:8192,BENCH_BLOCKS=80,WARMUP_BLOCKS=10 \
  /lustre/fsw/network_software_cloudai/snordmann/repos/_agent_playbook/flashinfer_opt12_20260702/paired_micro.sbatch
```

Use maximum-rank CUDA-event time per replay. Retain only if the paired direction
reproduces and Nsight shows fewer completion-flag loads or lower post-last-rank
combine time without a decode regression.

## Risks

- Tight polling may already hit a local cache, making backoff neutral or slower.
- `__nanosleep` duration is approximate and can overshoot; decode must be measured.
- The patch reduces poll frequency, not rank-arrival skew itself.
- Nsight system-load counts are needed to validate the proposed mechanism.

## Sources

- [One-sided kernel](../csrc/nv_internal/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.cu)
- [FlashInfer XQA bounded-wait precedent](../csrc/xqa/barriers.cuh)
- [FlashInfer persistent scheduler](../flashinfer/cute_dsl/attention/scheduler/persistent.py)
- [FlashInfer Blackwell CuTe/CUTLASS kernel](../flashinfer/gemm/kernels/cute_dsl/dense_gemm_bf16_fp4_blackwell.py)
- [CUDA nanosleep documentation and exponential-backoff example](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#nanosleep-function)
- [CUTLASS Blackwell CuTe tutorial](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/blackwell/01_mma_sm100.cu)
- [Prior one-sided experiment report](/Users/snordmann/Documents/DL%20Compilers/reports/2026-07-01-flashinfer-one-sided-optimization-experiments.md)
- [Routing follow-up](/Users/snordmann/Documents/DL%20Compilers/reports/2026-07-02-flashinfer-routing-decision-followup.md)
