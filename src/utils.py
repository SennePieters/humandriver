from __future__ import annotations
import ctypes
import platform
import random


class HiResTimer:
    """Context manager to request 1ms Windows system timer resolution.

    On Windows this reduces sleep jitter/latency for tighter timing loops.
    On other platforms, it is a no-op.
    """

    def __enter__(self):
        if ctypes and platform.system() == "Windows":
            ctypes.windll.winmm.timeBeginPeriod(1)
        return self

    def __exit__(self, exc_type, exc, tb):
        if ctypes and platform.system() == "Windows":
            ctypes.windll.winmm.timeEndPeriod(1)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Restrict value to [min_val, max_val]."""
    return max(min_val, min(max_val, value))


def random_uniform(a: float, b: float) -> float:
    """Return a random float between a and b, agnostic to order."""
    lo, hi = (a, b) if a <= b else (b, a)
    return random.uniform(lo, hi)
