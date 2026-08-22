"""
B0.4 — Measurement instrumentation for Ch5 runs.

Provides:
  - `cuda_sync_timer`: accurate GPU latency via torch.cuda.Event (falls back to
    wall-clock on CPU).
  - `peak_vram_mb`: peak allocated/reserved VRAM around a code block.
  - `run_guarded`: executes a thunk, classifying the outcome as ok / OOM / error so
    the feasibility frontier (OOM flag) is captured cleanly instead of crashing the
    whole sweep.
  - `git_commit`: records the commit hash of a repo for provenance.
"""

from __future__ import annotations

import contextlib
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch


def git_commit(repo_dir: str) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", repo_dir, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


@contextlib.contextmanager
def cuda_sync_timer(device: torch.device):
    """Context manager yielding a 1-element list whose [0] is filled with elapsed ms."""
    result: list[float] = [float("nan")]
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        start.record()
        try:
            yield result
        finally:
            end.record()
            torch.cuda.synchronize(device)
            result[0] = start.elapsed_time(end)  # ms
    else:
        t0 = time.perf_counter()
        try:
            yield result
        finally:
            result[0] = (time.perf_counter() - t0) * 1e3


def reset_peak_vram(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()


def peak_vram_mb(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {"peak_alloc_mb": 0.0, "peak_reserved_mb": 0.0}
    return {
        "peak_alloc_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
    }


def is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda oom" in msg


@dataclass
class RunOutcome:
    status: str                       # "ok" | "oom" | "error"
    value: Any = None
    error: str = ""
    metrics: dict[str, float] = field(default_factory=dict)


def run_guarded(fn: Callable[[], Any], device: torch.device) -> RunOutcome:
    """Run `fn`, classifying CUDA OOM vs other errors. On OOM, frees the cache so the
    next cell in a sweep starts clean."""
    try:
        value = fn()
        return RunOutcome(status="ok", value=value)
    except BaseException as exc:  # noqa: BLE001 - we want to catch OOM (BaseException subclass)
        if is_oom_error(exc):
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)
            return RunOutcome(status="oom", error=str(exc)[:300])
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return RunOutcome(status="error", error=f"{type(exc).__name__}: {exc}"[:500])
