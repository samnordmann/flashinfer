"""Host-only activation contract tests for NVFP4 MegaMoE."""

from __future__ import annotations

import dataclasses

import pytest

pytest.importorskip("flashinfer.moe_ep.kernel_src.cutedsl_megamoe")


def _config(**kwargs):
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import MegaMoENvfp4Config

    values = dict(
        rank=0,
        world_size=1,
        num_tokens_per_rank=64,
        num_topk=4,
        num_total_experts=4,
        hidden=2048,
        intermediate=2048,
    )
    values.update(kwargs)
    return MegaMoENvfp4Config(**values)


def test_swiglu_remains_the_default() -> None:
    config = _config()
    assert config.activation == "swiglu"
    assert config.intermediate_output == 1024


def test_relu2_preserves_the_physical_fc1_width() -> None:
    config = _config(activation="relu2", intermediate=1024)
    assert config.intermediate_output == 1024


def test_relu2_rejects_swiglu_clamp() -> None:
    with pytest.raises(ValueError, match="gate_up_clamp"):
        _config(activation="relu2", gate_up_clamp=10.0)


def test_unknown_activation_is_rejected() -> None:
    with pytest.raises(ValueError, match="activation"):
        _config(activation="gelu")


def test_backend_config_uses_one_fc1_projection_for_relu2() -> None:
    from flashinfer.moe_ep import (
        Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
    )

    config = Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=5120,
        top_k=22,
        activation="relu2",
    )
    assert config.fc1_projection_size == 5120


def test_backend_relu2_rejects_swiglu_clamp() -> None:
    from flashinfer.moe_ep import (
        Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
    )

    with pytest.raises(ValueError, match="only valid"):
        Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=5120,
            top_k=22,
            activation="relu2",
            activation_clamp=10.0,
        )


def test_profile_configs_only_change_tactic_fields() -> None:
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim.nvfp4 import (
        MegaMoENvfp4Frontend,
    )

    base = _config(num_tokens_per_rank=64, token_back_mode="reuse_dispatch_warps")
    small = dataclasses.replace(base, token_back_mode="epi_warps", flag_batch=4)
    frontend = MegaMoENvfp4Frontend(
        base,
        profile_configs=((32, small), (64, base)),
    )
    assert tuple(limit for limit, _ in frontend._profile_configs) == (32, 64)

    small_compiled = object()
    large_compiled = object()
    frontend._profile_megas = ((32, small_compiled), (64, large_compiled))
    assert frontend._select_compiled_profile(32) is small_compiled
    assert frontend._select_compiled_profile(33) is large_compiled
    assert frontend._select_compiled_profile(65) is None

    with pytest.raises(ValueError, match="protocol field"):
        MegaMoENvfp4Frontend(
            base,
            profile_configs=(
                (32, small),
                (64, dataclasses.replace(base, num_total_experts=8)),
            ),
        )
    with pytest.raises(ValueError, match="final profile limit"):
        MegaMoENvfp4Frontend(base, profile_configs=((32, small),))


def test_compatible_workspace_layout_reserves_union_slots() -> None:
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim.nvfp4 import (
        _apply_compatible_workspace_layout,
        _make_compatible_workspace_layout,
    )

    @dataclasses.dataclass(frozen=True)
    class Spec:
        name: str
        nbytes: int
        align: int = 16
        cute_dtype: str = "int32"
        shape: tuple[int, ...] = (1,)

    class FakeKernel:
        def __init__(self, local_specs, shared_specs):
            self._local_region_specs = local_specs
            self._shared_region_specs = shared_specs
            self._local_region_by_name = {spec.name: spec for spec in local_specs}
            self._shared_region_by_name = {spec.name: spec for spec in shared_specs}

    shared = [Spec("shared_counter", 8), Spec("src_token_topk_idx", 64)]
    small = FakeKernel(
        [
            Spec("counter", 8),
            Spec("l1_token_buffer", 64),
            Spec("nvlink_barrier_counter", 4),
        ],
        shared,
    )
    large = FakeKernel(
        [
            Spec("counter", 16),
            Spec("fc2_counter", 4),
            Spec("l1_token_buffer", 64),
            Spec("nvlink_barrier_counter", 4),
            Spec("fc2_output", 128),
        ],
        shared,
    )

    layout = _make_compatible_workspace_layout((small, large))
    _apply_compatible_workspace_layout(small, layout)
    _apply_compatible_workspace_layout(large, layout)

    assert small._local_offsets == large._local_offsets
    assert small._shared_offsets == large._shared_offsets
    assert layout.local_offsets["l1_token_buffer"] >= 32
    assert small.local_zero_i32_count == large.local_zero_i32_count
    assert small._local_total == large._local_total

    incompatible = FakeKernel(
        large._local_region_specs,
        [Spec("shared_counter", 16, shape=(2,)), shared[1]],
    )
    with pytest.raises(ValueError, match="shared workspace region"):
        _make_compatible_workspace_layout((small, incompatible))

    missing_shared_region = FakeKernel(large._local_region_specs, shared[1:])
    with pytest.raises(ValueError, match="same region names"):
        _make_compatible_workspace_layout((small, missing_shared_region))
