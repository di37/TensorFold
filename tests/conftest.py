"""Tests of the lane kernels need Metal 4 tensor units (M5-generation GPUs); elsewhere they are skipped."""

from __future__ import annotations

import importlib.util
import os

import pytest

# fp32 matmuls in fp32 on M5-generation GPUs, as GLM-5.3-Flash serves (its MLX_ENV); MLX reads this once a process
os.environ.setdefault("MLX_ENABLE_TF32", "0")
# no release checks or first-run notes from tests; tests/test_update.py turns them on where it tests them
os.environ.setdefault("TENSORFOLD_NO_UPDATE_CHECK", "1")

TENSOR_UNIT_TESTS = {
    "test_lane_qmm.py", "test_lane_attention.py", "test_lane_tree.py", "test_lane_fuse.py", "test_lane_glue_norm.py",
    "test_dflash_draft_vocab.py",
}


def _tensor_units() -> bool:
    try:
        from tensorfold.families.qwen3_5 import tensor_units

        return tensor_units()
    except Exception:  # noqa: BLE001 - no MLX or no Metal: no tensor units
        return False


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "torch: needs PyTorch (the CUDA backend's code); skipped where it isn't installed")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if importlib.util.find_spec("torch") is None:
        no_torch = pytest.mark.skip(reason="needs PyTorch (the CUDA backend's code)")
        for item in items:
            if item.get_closest_marker("torch") is not None:
                item.add_marker(no_torch)
    if _tensor_units():
        return
    skip = pytest.mark.skip(reason="needs Metal 4 tensor units (an M5-generation GPU)")
    for item in items:
        if item.path.name in TENSOR_UNIT_TESTS:
            item.add_marker(skip)


# MLX names the pipeline's limit when a launch passes it (M1/M2, a macOS VM's paravirtual GPU)
_THREADGROUP_LIMIT = "maximum allowed threads per threadgroup"
# M5 tensor-unit kernels include this header, which macOS 26 brings (CI runners and older Macs lack it)
_NO_TENSOR_HEADERS = "MetalPerformancePrimitives/MetalPerformancePrimitives.h' file not found"


def _environment_skip(item: pytest.Item, error: BaseException) -> str | None:
    """Why ``error`` is this machine's, not the code's: GLM's 1024-thread kernels where pipelines may take fewer, or
    tensor-unit kernels built on a macOS without their header. Elsewhere the error stays a failure."""

    text = str(error)
    if isinstance(error, RuntimeError) and _NO_TENSOR_HEADERS in text:
        return "tensor-unit kernels need macOS 26's MetalPerformancePrimitives"
    if isinstance(error, ValueError) and _THREADGROUP_LIMIT in text and "glm" in item.nodeid.lower():
        from tensorfold.kernels import threads

        if threads.probing:
            return (f"GLM-5.3-Flash serves on 256 GB Macs (M3 Ultra and later), whose pipelines take 1024 threads; "
                    f"{threads.chip()} took fewer")
    return None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: pytest.Item):
    outcome = yield
    error = outcome.excinfo[1] if outcome.excinfo else None
    reason = None if error is None else _environment_skip(item, error)
    if reason is not None:
        outcome.force_exception(pytest.skip.Exception(reason))
