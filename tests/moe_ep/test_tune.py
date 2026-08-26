# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the FlashInfer project

import sys
from types import ModuleType, SimpleNamespace

from flashinfer.moe_ep.tune import _parse_args, main


def _required_args() -> list[str]:
    return [
        "--hidden",
        "2048",
        "--intermediate",
        "5120",
        "--num-experts",
        "512",
        "--topk",
        "22",
        "--max-tokens",
        "16384",
    ]


def test_tune_defaults_to_swiglu() -> None:
    args = _parse_args(_required_args())

    assert args.activation == "swiglu"


def test_tune_accepts_nvfp4_relu2() -> None:
    args = _parse_args([*_required_args(), "--activation", "relu2"])

    assert args.dtype == "nvfp4"
    assert args.activation == "relu2"
    assert args.intermediate == 5120


def test_tune_rejects_relu2_for_mxfp8(capsys) -> None:
    result = main([*_required_args(), "--dtype", "mxfp8_e4m3", "--activation", "relu2"])

    assert result == 2
    assert "only wired for --dtype nvfp4" in capsys.readouterr().err


def test_tune_rejects_relu2_gate_up_clamp(capsys) -> None:
    result = main([*_required_args(), "--activation", "relu2", "--gate-up-clamp", "7"])

    assert result == 2
    assert "not valid with --activation relu2" in capsys.readouterr().err


def test_nvfp4_tuner_builds_and_resolves_relu2_geometry(monkeypatch) -> None:
    from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl import (
        tuner,
    )

    captured: dict = {}

    class _Workspace:
        def destroy(self) -> None:
            captured["destroyed"] = True

    def create_dummy(*args, **kwargs):
        captured["create_args"] = args
        captured["create_kwargs"] = kwargs
        return object(), object(), object(), _Workspace()

    def resolve_knobs(**kwargs):
        captured["resolve_kwargs"] = kwargs
        return {"flag_batch": 4}, "heuristic"

    fake_kernel = ModuleType("flashinfer.moe_ep.kernel_src.cutedsl_megamoe")
    fake_kernel.COMBINE_FORMAT_NAMES = {"bf16": "bf16"}
    fake_kernel.autotune_nvfp4_mega_moe = object()
    fake_kernel.create_dummy_nvfp4_inputs = create_dummy
    fake_kernel.nvfp4_candidates = lambda **kwargs: [{}]
    fake_kernel.resolve_knobs = resolve_knobs
    monkeypatch.setitem(
        sys.modules,
        "flashinfer.moe_ep.kernel_src.cutedsl_megamoe",
        fake_kernel,
    )
    monkeypatch.setattr(
        tuner,
        "finish_sweep",
        lambda *args, **kwargs: {"status": "ok"},
    )

    args = SimpleNamespace(
        live_tokens=4,
        intermediate=5120,
        activation="relu2",
        num_experts=512,
        topk=22,
        hidden=2048,
        gate_up_clamp=None,
        combine_dtype="bf16",
        seed=0,
        allow_nondeterministic=False,
        sweep="schedule",
        base_knobs=None,
        dtype="nvfp4",
    )
    result = tuner.tune_one(args, rank=0, world_size=4, max_tokens=16384)

    assert result == {"status": "ok"}
    assert captured["create_args"][7] == 5120
    assert captured["create_kwargs"]["activation"] == "relu2"
    assert captured["resolve_kwargs"]["dtype"] == "nvfp4_relu2"
    assert captured["resolve_kwargs"]["intermediate"] == 5120
    assert captured["destroyed"]
