from __future__ import annotations
import math
from typing import List
from .telemetry import recorder


def summarize_speeds() -> str:
    """Summarize instantaneous movement speeds from recorded 'move' events.

    Computes per-step speed as distance / dt_ms, with a 2ms floor on dt for stability.
    Reports average, p95, p99, max and sample count.
    """
    move_events = [event for event in recorder.events if event.kind == "move"]
    if len(move_events) < 2:
        return "No move data"
    speeds_px_per_ms: List[float] = []
    for i in range(1, len(move_events)):
        delta_x = move_events[i].x - move_events[i - 1].x
        delta_y = move_events[i].y - move_events[i - 1].y
        dt_ms = max(9.0, (move_events[i].t - move_events[i - 1].t) * 1000.0)
        speeds_px_per_ms.append(math.hypot(delta_x, delta_y) / dt_ms)
    if not speeds_px_per_ms:
        return "No speeds"
    speeds_sorted = sorted(speeds_px_per_ms)
    n = len(speeds_sorted)
    average = sum(speeds_px_per_ms) / n
    p95 = speeds_sorted[int(0.95 * n) - 1]
    p99 = speeds_sorted[int(0.99 * n) - 1]
    return (
        f"speed px/ms: avg={average:.3f}, p95={p95:.3f}, p99={p99:.3f}, "
        f"max={max(speeds_px_per_ms):.3f}, samples={n}"
    )
