"""Host-only activation contract tests for NVFP4 MegaMoE."""

from __future__ import annotations

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
