from __future__ import annotations
import asyncio
import math
import random
import time
import logging
from typing import List, Dict, Tuple, Optional, Callable, Awaitable, Any
from zendriver import cdp

from .config import cfg
from .telemetry import recorder
from .geometry import get_viewport, _clamp_point_to_viewport

# Config values copied locally for speed/readability
EASE_POWER: float = getattr(cfg, "EASE_POWER", 2.2)
COALESCE_THRESHOLD_PX: float = getattr(cfg, "COALESCE_THRESHOLD_PX", 1.8)

MIN_SPEED_PX_PER_MS: float = getattr(cfg, "MIN_SPEED_PX_PER_MS", 0.35)
MAX_SPEED_PX_PER_MS: float = getattr(cfg, "MAX_SPEED_PX_PER_MS", 1.60)
TIMING_JITTER_S: float = getattr(cfg, "TIMING_JITTER_S", 0.0035)
MIN_SLEEP_S: float = getattr(cfg, "MIN_SLEEP_S", 0.003)
TARGET_HZ: float = getattr(cfg, "TARGET_HZ", 110)


CDP_SEND_TIMEOUT_S: float = 0.05
CDP_SEND_MIN_INTERVAL_S: float = 0.015

_last_cdp_send_timestamp: Optional[float] = None
_CDP_SEND_SEMAPHORE = asyncio.Semaphore(1)
_MOUSE_SEND_LOCK = asyncio.Lock()


async def _send_cdp_event(
    page, fn: Callable[[], Awaitable[Any]], *, label: str
) -> None:
    """Bounded-time CDP send with shielded task."""
    logger = logging.getLogger(__name__)
    global _last_cdp_send_timestamp
    async with _CDP_SEND_SEMAPHORE:
        now = time.perf_counter()
        if _last_cdp_send_timestamp is not None:
            gap = now - _last_cdp_send_timestamp
            if gap < CDP_SEND_MIN_INTERVAL_S:
                await asyncio.sleep(CDP_SEND_MIN_INTERVAL_S - gap)

        send_task = asyncio.create_task(fn())
        start = time.perf_counter()
        try:
            await asyncio.wait_for(
                asyncio.shield(send_task), timeout=CDP_SEND_TIMEOUT_S
            )
            _last_cdp_send_timestamp = time.perf_counter()
        except asyncio.TimeoutError:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            logger.warning(
                "CDP %s pending %.1f ms (>%.0f ms); letting it finish in background",
                label,
                elapsed_ms,
                CDP_SEND_TIMEOUT_S * 1000.0,
            )

            def _late_log(task: asyncio.Task) -> None:
                try:
                    task.result()
                    global _last_cdp_send_timestamp
                    _last_cdp_send_timestamp = time.perf_counter()
                except Exception:
                    pass

            send_task.add_done_callback(_late_log)
            _last_cdp_send_timestamp = time.perf_counter()
        except Exception:
            logger.warning("CDP %s failed (skipped this event)", label, exc_info=True)


def windmouse(
    start: Tuple[float, float],
    target: Tuple[float, float],
    *,
    wind: float = cfg.WIND,
    gravity: float = cfg.GRAVITY,
    min_step: float = cfg.MIN_STEP,
    max_step: float = cfg.MAX_STEP,
    target_area: float = cfg.TARGET_AREA,
    jitter: float = cfg.JITTER,
) -> List[Tuple[float, float]]:
    """Simple windmouse path generator."""
    try:
        sx, sy = float(start[0]), float(start[1])
        tx, ty = float(target[0]), float(target[1])
        wind = float(wind)
        gravity = float(gravity)
        min_step = float(min_step)
        max_step = float(max_step)
        target_area = float(target_area)
        jitter = float(jitter)
    except Exception:
        return [start, target]

    vx = vy = 0.0
    wind_x = wind_y = 0.0
    path: List[Tuple[float, float]] = []
    last_dist = math.hypot(tx - sx, ty - sy)
    max_iters = 2000

    for i in range(max_iters):
        dist = math.hypot(tx - sx, ty - sy)
        if dist < 1.0:
            break
        if i > 25 and dist >= last_dist:
            break

        wind_mag = min(wind, dist)
        if dist >= target_area:
            wind_x = wind_x / math.sqrt(3) + (
                random.random() * wind_mag * 2 - wind_mag
            ) / math.sqrt(5)
            wind_y = wind_y / math.sqrt(3) + (
                random.random() * wind_mag * 2 - wind_mag
            ) / math.sqrt(5)
        else:
            wind_x /= math.sqrt(3)
            wind_y /= math.sqrt(3)

        vx += wind_x + gravity * (tx - sx) / max(dist, 1e-6)
        vy += wind_y + gravity * (ty - sy) / max(dist, 1e-6)

        speed = math.hypot(vx, vy)
        if speed > max_step:
            scale = max_step / speed
            vx *= scale
            vy *= scale

        sx += vx
        sy += vy
        path.append(
            (sx + random.uniform(-jitter, jitter), sy + random.uniform(-jitter, jitter))
        )
        last_dist = dist

    if not path or path[-1] != (tx, ty):
        path.append((tx, ty))
    return path


async def _emit_mouse_move(page, x: float, y: float) -> None:
    """Minimal mouse move emit with fast retries."""
    recorder.log_move(x, y)
    if cdp is not None:
        last_err: Optional[BaseException] = None
        async with _MOUSE_SEND_LOCK:
            for attempt in range(3):
                try:
                    task = asyncio.create_task(
                        page.send(
                            cdp.input_.dispatch_mouse_event(
                                type_="mouseMoved", x=float(x), y=float(y)
                            )
                        )
                    )
                    resp = await asyncio.wait_for(asyncio.shield(task), timeout=0.10)
                    if isinstance(resp, dict) and resp.get("success") is False:
                        raise RuntimeError(
                            "CDP dispatchMouseEvent returned success=False"
                        )
                    return
                except asyncio.TimeoutError:
                    return
                except Exception as exc:
                    last_err = exc
                await asyncio.sleep(0.02)
        raise RuntimeError(
            f"mouseMoved send failed at ({x:.2f}, {y:.2f})"
        ) from last_err


async def dispatch_mouse_path(
    page,
    points: List[Dict[str, float]],
    target_width_pixels: Optional[float] = None,
) -> Tuple[float, float]:
    """Simplified windmouse-only dispatcher."""
    viewport_width, viewport_height = await get_viewport(page)
    if not points:
        current = getattr(
            page, "_mouse_pos", (viewport_width / 2.0, viewport_height / 2.0)
        )
        return current

    pts = [
        {
            "x": max(0.0, min(viewport_width, float(p["x"]))),
            "y": max(0.0, min(viewport_height, float(p["y"]))),
        }
        for p in points
    ]

    current = getattr(page, "_mouse_pos", None)
    if current is not None:
        current = _clamp_point_to_viewport(
            current[0], current[1], viewport_width, viewport_height
        )
    else:
        current = (pts[0]["x"], pts[0]["y"])

    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    if current != (pts[0]["x"], pts[0]["y"]):
        segments.append((current, (pts[0]["x"], pts[0]["y"])))
    for i in range(len(pts) - 1):
        segments.append(
            ((pts[i]["x"], pts[i]["y"]), (pts[i + 1]["x"], pts[i + 1]["y"]))
        )

    interval = float(getattr(cfg, "MOVE_INTERVAL_S", 0.015))
    if interval > 0.1:
        interval = 0.015
    max_move_duration = float(getattr(cfg, "MAX_MOVE_DURATION_S", 1.0))

    for start, end in segments:
        try:
            path_points = windmouse(start, end)
        except Exception:
            path_points = [start, end]
        if not path_points:
            path_points = [start, end]

        if len(path_points) > 1200:
            stride = max(1, len(path_points) // 1200)
            path_points = path_points[::stride] + [path_points[-1]]

        step_interval = interval
        if max_move_duration > 0 and len(path_points) > 0:
            step_interval = min(
                interval, max(0.001, max_move_duration / len(path_points))
            )

        for x, y in path_points:
            x, y = _clamp_point_to_viewport(x, y, viewport_width, viewport_height)
            await _emit_mouse_move(page, x, y)
            if step_interval > 0:
                await asyncio.sleep(step_interval)

    final_x, final_y = segments[-1][1] if segments else (pts[-1]["x"], pts[-1]["y"])
    page._mouse_pos = (final_x, final_y)
    return final_x, final_y
