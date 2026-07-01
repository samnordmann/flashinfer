# MoE Dispatch Payload Flattening

Date: 2026-07-01
Repo: FlashInfer v0.6.11 direct-output stack
Branch / base: `codex/fi-v0611-payload-flatten` / `647ad017`
Target: Ultra NVFP4, H=8192, top-k=22, 512 experts, EP4, CUDA graphs

## Hypothesis

For EP4, at most four of 22 routed destinations are live. Caching those compact destinations and
all payload/destination base pointers once per token tile removes repeated lane-local address work.
Flattening payloads with the same safe vector width combines the two 88-byte payloads into one
22-unit queue instead of two 11-lane phases.

## Change

The compact path is selected for multiple payloads with EP <= 8. It preserves payload boundaries,
chooses vector width from byte count and actual pointer alignment, and leaves the original dispatch
path unchanged for a single payload or larger EP.

## Correctness

- Added multi-rank dispatch coverage for `top_k=22`.
- `git diff --check` and Python syntax checks pass.
- A CPU queue model passed 1,000 randomized size/alignment/fanout cases.
- No CUDA compiler or GPU is available locally; CUDA build and MNNVL tests are not run.

## Benchmark

Run baseline and candidate in the same target image and node allocation:

```bash
mpirun -np 4 python benchmarks/flashinfer_benchmark.py \
  --routine moe_a2a_dispatch_combine --num_tokens 1 \
  --hidden_size 8192 --num_experts 512 --top_k 22 \
  --quant_dtype nvfp4 --validate
```

Sweep `--num_tokens` and `--max_num_tokens`; keep CUDA graphs enabled for the target comparison and
repeat with `--no_cuda_graph`. Use `--per_phase_timing` only for dispatch attribution because it
disables CUDA graphs.

## Sources

- `csrc/nv_internal/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.cu`
- `benchmarks/routines/moe_comm.py`
- `tests/comm/test_mnnvl_moe_alltoall.py`
