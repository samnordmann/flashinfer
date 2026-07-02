# NVFP4 H8192 Dispatch Payload Specialization

Date: 2026-07-02
Repo: FlashInfer
Branch: `snordmann/fi-opt12-02-nvfp4-payload-20260702`
Base: `04bf64bd`
Target: 1x GB200 node, TP1/DP4/EP4, 512 experts, top-k 22, hidden size 8192
CUDA graph: required for correctness and performance validation

## Hypothesis

The target dispatch always carries four ordered payloads with byte widths
`[4096, 512, 88, 88]`. Selecting one fixed-layout kernel from the validated runtime descriptors
removes the device-side payload loop and per-payload vector-width branches. The likely benefit is
lower dispatch instruction and register overhead; mandatory source loads and remote stores are
unchanged.

## Changes

- Add one extra `top_k=22` kernel instantiation for EP4, 512 experts, and descriptors
  `[(1, 4096), (1, 512), (4, 22), (4, 22)]`.
- Use fixed vector widths `[16, 16, 8, 8]` in that kernel.
- Preserve the existing generic kernel for every other descriptor or target shape.
- Add an EP4 unit case for the exact target payload.

Source: [dispatch kernel](../csrc/nv_internal/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.cu),
[MNNVL test](../tests/comm/test_mnnvl_moe_alltoall.py).

## Correctness

The specialized path retains payload order, destination indexing, routing metadata, barriers,
publication, and completion polling. It changes only how the four byte-copy loops are selected.
The host predicate depends only on call descriptors, so capture records a fixed kernel launch and
replay reads no new host or device state.

Repository unit check in a FlashInfer development environment with MPI and `mpi4py`:

```bash
mpirun -np 4 pytest -q \
  tests/comm/test_mnnvl_moe_alltoall.py::test_moe_a2a_dispatch_nvfp4_h8192_payload
```

The official target image lacks MPI tooling. There, use the established `torchrun`/Gloo EP4
correctness harness, changed to the exact four payloads above, with zero-token/asymmetric ranks and
at least 1,000 CUDA graph replays. Do not reuse its default H256 BF16 payload because that selects
the generic fallback.

## Performance

No GPU performance result is available yet. Treat this as a benchmark candidate, not a measured
optimization.

Use the same-process paired harness shape from
[`route_owner_same_process_bench.py`](/Users/snordmann/Documents/DL%20Compilers/lyris_jobs/flashinfer_route_owner_decision_20260702/route_owner_same_process_bench.py),
but capture the generic and fixed-payload kernels from one instrumented module. Use EP4, 512
experts, top-k 22, ordered payloads `[uint8[N,4096], uint8[N,512], int32[N,22],
float32[N,22]]`, CUDA graphs, alternating order, 80 paired blocks, and local-token counts
`1,4,16,64,512,4096,32768`. Report maximum-rank replay latency and paired confidence intervals.

## Profiling Evidence

The fresh target profile reports 183 registers/thread for dispatch and identifies dispatch as the
largest communication kernel, but most mixed-window latency is rank-arrival skew. Confirm any
microbenchmark gain with register counts and post-last-rank dispatch time before an end-to-end run.

## Risks

- The extra kernel may increase module size and JIT time.
- Static calls may not reduce registers enough to change occupancy or critical-path latency.
- A kernel-local gain can be hidden by rank skew and may not improve high-throughput end to end.
- Descriptor order changes intentionally fall back to the generic kernel.

## Sources

- [Prior one-sided experiments](/Users/snordmann/Documents/DL%20Compilers/reports/2026-07-01-flashinfer-one-sided-optimization-experiments.md)
- [Routing decision follow-up](/Users/snordmann/Documents/DL%20Compilers/reports/2026-07-02-flashinfer-routing-decision-followup.md)
