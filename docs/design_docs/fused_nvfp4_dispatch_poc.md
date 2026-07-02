# Fused NVFP4 MoE Dispatch POC

Date: 2026-07-02
Base: `04bf64bd`
Target: GB200 EP4, BF16 H8192, NVFP4, 512 experts, top-k 22

## Hypothesis

Fuse BF16-to-NVFP4 quantization into one-sided dispatch so packed activations
and block scales go directly to destination workspace slots. This removes the
local quantized tensors from the producer-consumer boundary.

## Scope

- New opt-in `moe_a2a_dispatch_nvfp4` API and `MoeAlltoAll.dispatch_nvfp4` method.
- BF16 input, scalar FP32 global scale, 16-value NVFP4 blocks, linear E4M3
  scales, SM100+.
- Generated outputs are packed activation, scales, then up to four ordinary
  pass-through payloads.
- Generic dispatch and combine code paths are unchanged.
- `FLASHINFER_NVFP4_4OVER6=1` is rejected.

The fused kernel preserves the TOP_K-wide global routing metadata and duplicate
positions consumed by combine. A private compact destination list is used only
for dispatch stores.

## Dataflow

For H8192, standalone quantization materializes 4,096 activation bytes and 512
scale bytes per source token. Fusion removes exactly:

```text
4,608 B local quantization writes + 4,608 B dispatch reads = 9,216 B/token
```

The target's expert-ID and weight payloads total 176 B/token. This POC reloads
those small payloads once per unique destination instead of loading once and
fanning out. With `U` unique destinations, logical local traffic therefore
falls by `9,216 - 176 * (U - 1)` B/token: 9,216 B at U=1, 9,040 B at U=2, and
8,688 B at U=4. Cache behavior can make physical HBM traffic differ.

MNNVL writes are unchanged at `U * 4,784 B/token` for the target payload. The
BF16 source read (16,384 B/token) and quantization arithmetic remain mandatory.
The graph loses the standalone quantization launch/node.

## Correctness

Added tests compare fused output byte-for-byte with:

```python
packed, scales = nvfp4_quantize(x, global_scale, sfLayout=SfLayout.layout_linear)
received = moe_a2a_dispatch(routes, [packed, scales, routes, weights, token_ids], ...)
```

The candidate comparison reorders received rows by token ID, checks packed
bytes, scales, pass-through payloads, receive counts, and persistent target-rank
metadata. A separate test captures the fused call and replays it 16 times while
changing BF16 input.

These GPU tests were not run in this bounded pass.

## Benchmark Signature

Capture two graphs over identical persistent inputs and MNNVL allocations:

```python
# Baseline
packed, scales = nvfp4_quantize(
    x, global_scale, sfLayout=SfLayout.layout_linear, enable_pdl=False
)
moe_a2a_dispatch(routes, [packed, scales, routes, weights], workspace, ...)

# Candidate
moe_a2a_dispatch_nvfp4(
    x, global_scale, routes, [routes, weights], workspace, ...
)
```

Use EP4, H8192, top-k 22, 512 experts; token counts 4, 64, 512, 4096, and
32768; 5 warmup blocks and at least 80 alternating paired blocks. Report the
maximum-rank CUDA-event latency, median paired ratio, bootstrap confidence
interval, registers, spills, occupancy, and graph replay correctness. Run
target vLLM AB/BA only if the same-process kernel result is positive.

## Performance

Unmeasured. No speedup is claimed.

## Risks

- CUDA compilation and GB200 execution are unverified.
- Fusing quantization may increase dispatch register pressure or reduce
  occupancy; inspect ptxas output before performance conclusions.
- Private compact routing incorporates a previously neutral-at-e2e dispatch
  technique, so profiling must separate fusion savings from routing effects.
- Only standard NVFP4 is supported; 4-over-6 and row-wise/inverse global scales
  require additional specializations.
- The existing generation-counter wrap and one-sided proxy-alias assumptions
  are inherited unchanged.
