# EP4 Dispatch-Only Route Compaction

Date: 2026-07-01
Base: `647ad0176825b014a0117c01010ca5e31f2203b3`
Branch: `codex/fi-v0611-route-dispatch-only`

## Hypothesis

For EP4/top-k22, a token can target at most four ranks. Dispatch does not need 22 shared-memory
route entries, 22 register pairs, or 22 destination pointers per thread to copy each payload.

## Change

- Preserve the global `[token, top_k]` target-rank and send-index tensors exactly: the first expert
  for a rank remains at its original top-k position, and later experts for that rank remain `-1`.
- For EP4 with `top_k > 4`, lane 0 also builds a dense four-entry route in shared memory. Payload
  fanout consumes only that private route.
- Keep the generic EP path and the complete combine implementation unchanged.
- Keep the public API, workspace layout, synchronization protocol, and launch topology unchanged.

## Expected Effect

Retained from the earlier compact-route prototype:

- EP4 dispatch shared route state falls from 22 to 4 rank/index pairs per token.
- Each dispatch thread holds at most four route entries and four destination pointers.
- Payload fanout examines at most four destinations.
- Only the producer lane scans top-k routing metadata on the specialized path.

Intentionally lost to preserve bitwise combine association:

- Combine still loads the sparse 22-entry global metadata.
- Combine still owns 22 accumulators and uses the baseline 22-leaf BF16 reduction tree.
- Global route metadata still performs 22 rank writes and 22 slot writes per token.

No performance gain is claimed before GB200 measurement.

## Correctness

`test_moe_a2a_dispatch_ep4_topk22_metadata` gives each EP4 source rank a distinct top-k22 route
with repeated destination ranks. One token per source makes the baseline slot contract
deterministic, allowing exact comparison of all 22 target-rank and send-index entries. The existing
payload and counter verification is then applied to the same dispatch.

Local checks:

- `python3 -m pre_commit run --files <kernel> <test>`: passed.
- `python3 -m py_compile tests/comm/test_mnnvl_moe_alltoall.py`: passed.
- Source comparison from the first combine template to EOF against `647ad017`: identical.
- GPU/MNNVL execution: not run locally; this host has no PyTorch, MPI, or CUDA compiler/runtime.

## Sources

- Kernel: `csrc/nv_internal/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.cu`
- Test: `tests/comm/test_mnnvl_moe_alltoall.py`
