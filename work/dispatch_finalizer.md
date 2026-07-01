# Dispatch Finalizer Candidate

Date: 2026-07-01
Base: `647ad0176825b014a0117c01010ca5e31f2203b3`
Target: Nemotron Ultra NVFP4, H8192, top-k 22, 512 experts, EP4, GB200

## Hypothesis

The one-block-per-token dispatch path performs one contended
`local_token_counter` atomic per token so the last completed token CTA can
publish counters and completion flags. A separate one-warp finalizer ordered
after dispatch can remove those atomics. This is most likely to help large
prefill batches; its extra launch may offset the gain for small decode batches.

## Semantic Contract

- The data kernel still produces identical payload, routing, and send-counter
  data.
- Same-stream launch ordering ensures all local dispatch CTAs finish before
  finalization starts.
- "Deterministic" applies to finalizer sequencing; existing atomic assignment
  of per-destination payload slots is intentionally unchanged.
- The finalizer preserves the existing remote `recv_counters`, system release
  fence, completion-flag publication, peer polling, and timeout behavior.
- Zero-token ranks launch the finalizer and therefore still participate in the
  cross-rank protocol.
- Python, FFI, workspace layout, and returned tensors are unchanged.
- CUDA graph capture remains static and allocation-free: the captured sequence
  gains one fixed kernel launch and has no host synchronization.

## Validation

- Added a distributed dispatch case with asymmetric rank-local token counts,
  one empty rank, and target top-k 22.
- `python3 -m compileall -q tests/comm/test_mnnvl_moe_alltoall.py`: passed.
- Pre-commit on changed files: passed.
- Distributed MNNVL correctness, CUDA graph replay, and performance were not
  run locally because this host has no CUDA/MNNVL environment. No Lyris job was
  submitted, per request.

## Risk

The candidate adds one kernel launch per dispatch. It should only be retained
if same-node EP4 testing shows that removing the per-token atomic exceeds that
launch cost and improves the target end-to-end workload, not only a large-token
microbenchmark.
