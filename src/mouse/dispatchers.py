from __future__ import annotations
import asyncio
import math
import random
import time
import bisect
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable, Awaitable, Any
from zendriver import cdp

from .config import cfg
from .telemetry import get_mouse_lock, get_mouse_recorder
from .geometry import get_viewport, _clamp_point_to_viewport

# Config values copied locally for speed/readability
EASE_POWER: float = getattr(cfg, "EASE_POWER", 2.2)
COALESCE_THRESHOLD_PX: float = getattr(cfg, "COALESCE_THRESHOLD_PX", 1.8)

MIN_SPEED_PX_PER_MS: float = getattr(cfg, "MIN_SPEED_PX_PER_MS", 0.35)
MAX_SPEED_PX_PER_MS: float = getattr(cfg, "MAX_SPEED_PX_PER_MS", 1.60)
TIMING_JITTER_S: float = getattr(cfg, "TIMING_JITTER_S", 0.0035)
MIN_SLEEP_S: float = getattr(cfg, "MIN_SLEEP_S", 0.003)
TARGET_HZ: float = getattr(cfg, "TARGET_HZ", 110)

BRIDGE_THRESHOLD_PX: float = getattr(cfg, "BRIDGE_THRESHOLD_PX", 6.0)
BRIDGE_STEPS_MINMAX: Tuple[int, int] = getattr(cfg, "BRIDGE_STEPS_MINMAX", (8, 18))
BRIDGE_CURVE_JITTER_FRAC: float = getattr(cfg, "BRIDGE_CURVE_JITTER_FRAC", 0.06)

SMOOTH_ITERS: int = getattr(cfg, "SMOOTH_ITERS", 1)  # 0 = off
SMOOTH_WEIGHT: float = getattr(cfg, "SMOOTH_WEIGHT", 0.22)

GLOBAL_MIN_INTERVAL_S: float = getattr(cfg, "GLOBAL_MIN_INTERVAL_S", 0.006)
FINAL_SNAP_EPS_PX: float = getattr(cfg, "FINAL_SNAP_EPS_PX", 0.4)

CURVE_MIN_WAYPOINTS: int = getattr(cfg, "CURVE_MIN_WAYPOINTS", 9)
CURVE_MAX_WAYPOINTS: int = getattr(cfg, "CURVE_MAX_WAYPOINTS", 14)
CURVE_LATERAL_FRAC_MIN: float = getattr(cfg, "CURVE_LATERAL_FRAC_MIN", 0.08)
CURVE_LATERAL_FRAC_MAX: float = getattr(cfg, "CURVE_LATERAL_FRAC_MAX", 0.18)
CURVE_ALONG_JITTER_FRAC: float = getattr(cfg, "CURVE_ALONG_JITTER_FRAC", 0.03)

WOBBLE_LATERAL_FRAC: float = getattr(cfg, "WOBBLE_LATERAL_FRAC", 0.04)
WOBBLE_FREQ_1: float = getattr(cfg, "WOBBLE_FREQ_1", 0.8)
WOBBLE_FREQ_2: float = getattr(cfg, "WOBBLE_FREQ_2", 2.2)

END_JITTER_FRAC: float = getattr(cfg, "END_JITTER_FRAC", 0.006)

CDP_SEND_TIMEOUT_S: float = 0.05  # fail-fast to avoid stalling the event loop
CDP_SEND_MIN_INTERVAL_S: float = (
    0.015  # throttle CDP sends to browser scan rate (~15ms)
)


@dataclass
class _PageCDPState:
    semaphore: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(1))
    last_send_ts: Optional[float] = None


def _get_cdp_state(page) -> _PageCDPState:
    if not hasattr(page, "_humandriver_cdp_state"):
        page._humandriver_cdp_state = _PageCDPState()
    return page._humandriver_cdp_state


async def _send_cdp_event(
    page, fn: Callable[[], Awaitable[Any]], *, label: str
) -> None:
    """Bounded-time CDP send with shielded task so it can't be cancelled mid-flight."""
    logger = logging.getLogger(__name__)
    state = _get_cdp_state(page)

    async with state.semaphore:
        now = time.perf_counter()
        if state.last_send_ts is not None:
            gap = now - state.last_send_ts
            if gap < CDP_SEND_MIN_INTERVAL_S:
                await asyncio.sleep(CDP_SEND_MIN_INTERVAL_S - gap)

        send_task = asyncio.create_task(fn())
        start = time.perf_counter()
        try:
            await asyncio.wait_for(
                asyncio.shield(send_task), timeout=CDP_SEND_TIMEOUT_S
            )
            state.last_send_ts = time.perf_counter()
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
                    state.last_send_ts = time.perf_counter()
                    logger.debug(
                        "CDP %s completed late after %.1f ms",
                        label,
                        (time.perf_counter() - start) * 1000.0,
                    )
                except Exception:
                    logger.warning("CDP %s failed after timeout", label, exc_info=True)

            send_task.add_done_callback(_late_log)
            state.last_send_ts = time.perf_counter()
        except Exception:
            logger.warning("CDP %s failed (skipped this event)", label, exc_info=True)


def _smoothstep_01(t: float) -> float:
    """Cubic smoothstep interpolation in [0,1]."""
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def _choose_average_speed_px_per_ms(total_distance_pixels: float) -> float:
    """Pick an average speed (px/ms) based on total distance (heuristic)."""
    normalized_distance = min(1.5, total_distance_pixels / 1200.0)
    base = max(0.35, MIN_SPEED_PX_PER_MS + 0.05)
    span = max(0.55, (MAX_SPEED_PX_PER_MS - 0.20))
    speed = base + span * _smoothstep_01(min(1.0, normalized_distance))
    return max(MIN_SPEED_PX_PER_MS, min(MAX_SPEED_PX_PER_MS, speed))


def _estimate_duration_fitts_style(
    distance_pixels: float,
    target_width_pixels: float,
    constant_a_ms: float = 120.0,
    constant_b_ms: float = 180.0,
) -> float:
    """Estimate movement duration using a Fitts' law–style model (seconds)."""
    target_width_pixels = max(4.0, float(target_width_pixels))
    distance_pixels = max(0.0, float(distance_pixels))
    movement_time_ms = constant_a_ms + constant_b_ms * math.log2(
        distance_pixels / target_width_pixels + 1.0
    )
    movement_time_ms *= random.uniform(0.95, 1.08)
    return max(0.06, movement_time_ms / 1000.0)


def _ease_fraction_symmetric(
    timeline_fraction: float, power: float = EASE_POWER
) -> float:
    """Symmetric ease-in/ease-out mapping of t∈[0,1] with a tunable exponent."""
    t = max(0.0, min(1.0, timeline_fraction))
    up, down = t**power, (1.0 - t) ** power
    return up / (up + down) if (up + down) > 0 else t


def _coalesce_small_steps(
    points: List[Dict[str, float]], threshold_pixels: float
) -> List[Dict[str, float]]:
    """Merge very small consecutive steps to avoid over-sampling (reduces noise/overhead)."""
    if not points:
        return points
    result: List[Dict[str, float]] = [points[0]]
    accumulated_dx = 0.0
    accumulated_dy = 0.0

    for i in range(1, len(points)):
        delta_x = points[i]["x"] - result[-1]["x"]
        delta_y = points[i]["y"] - result[-1]["y"]
        step_length = math.hypot(delta_x, delta_y)
        if step_length < threshold_pixels:
            accumulated_dx += delta_x
            accumulated_dy += delta_y
            if i == len(points) - 1:
                result.append(
                    {
                        "x": result[-1]["x"] + accumulated_dx,
                        "y": result[-1]["y"] + accumulated_dy,
                    }
                )
        else:
            if abs(accumulated_dx) + abs(accumulated_dy) > 0:
                result.append(
                    {
                        "x": result[-1]["x"] + accumulated_dx,
                        "y": result[-1]["y"] + accumulated_dy,
                    }
                )
                accumulated_dx = accumulated_dy = 0.0
            result.append(points[i])
    return result


def _chaikin_corner_cutting(
    points: List[Dict[str, float]], weight: float
) -> List[Dict[str, float]]:
    """One pass of Chaikin corner-cutting smoothing. Preserves endpoints."""
    if len(points) < 3:
        return points[:]
    smoothed: List[Dict[str, float]] = [points[0]]
    for i in range(0, len(points) - 1):
        p0, p1 = points[i], points[i + 1]
        q = {
            "x": (1 - weight) * p0["x"] + weight * p1["x"],
            "y": (1 - weight) * p0["y"] + weight * p1["y"],
        }
        r = {
            "x": weight * p0["x"] + (1 - weight) * p1["x"],
            "y": weight * p0["y"] + (1 - weight) * p1["y"],
        }
        smoothed.append(q)
        smoothed.append(r)
    smoothed.append(points[-1])
    return smoothed


def _build_bridge_waypoints(
    start_point: Tuple[float, float],
    end_point: Tuple[float, float],
) -> List[Dict[str, float]]:
    """Generate a short, curvy bridging path between two positions."""
    start_x, start_y = start_point
    end_x, end_y = end_point
    count = random.randint(*BRIDGE_STEPS_MINMAX)

    delta_x, delta_y = end_x - start_x, end_y - start_y
    distance = math.hypot(delta_x, delta_y) or 1.0
    tangent_x, tangent_y = delta_x / distance, delta_y / distance
    normal_x, normal_y = -tangent_y, tangent_x

    jitter_max = getattr(cfg, "BRIDGE_CURVE_JITTER_FRAC", 0.06)
    jitter_max = max(0.10, min(0.18, jitter_max * 2.0))
    anchor_fractions = [0.25, 0.55, 0.85]
    anchors: List[Tuple[float, float]] = []
    for fraction in anchor_fractions:
        anchor_base_x = start_x + delta_x * fraction
        anchor_base_y = start_y + delta_y * fraction
        lateral_offset = random.uniform(-jitter_max, jitter_max) * distance
        forward_offset = random.uniform(-0.02, 0.04) * distance
        anchors.append(
            (
                anchor_base_x + normal_x * lateral_offset + tangent_x * forward_offset,
                anchor_base_y + normal_y * lateral_offset + tangent_y * forward_offset,
            )
        )

    polyline = [(start_x, start_y), *anchors, (end_x, end_y)]

    cumulative_lengths = [0.0]
    total_length = 0.0
    for i in range(1, len(polyline)):
        segment_length = math.hypot(
            polyline[i][0] - polyline[i - 1][0], polyline[i][1] - polyline[i - 1][1]
        )
        total_length += segment_length
        cumulative_lengths.append(total_length)
    if total_length <= 1e-6:
        return [{"x": end_x, "y": end_y}]

    points: List[Dict[str, float]] = [{"x": start_x, "y": start_y}]
    for i in range(1, count):
        t = i / count
        s = t * total_length
        j = bisect.bisect_left(cumulative_lengths, s)
        if j <= 0:
            x, y = polyline[0]
        elif j >= len(polyline):
            x, y = polyline[-1]
        else:
            s0, s1 = cumulative_lengths[j - 1], cumulative_lengths[j]
            p0, p1 = polyline[j - 1], polyline[j]
            u = 0.0 if s1 <= s0 else (s - s0) / (s1 - s0)
            x = p0[0] + (p1[0] - p0[0]) * u
            y = p0[1] + (p1[1] - p0[1]) * u
        points.append({"x": x, "y": y})
    points.append({"x": end_x, "y": end_y})
    return points


def _make_curvy_polyline(
    start_point: Tuple[float, float],
    end_point: Tuple[float, float],
    viewport_width: float,
    viewport_height: float,
) -> List[Dict[str, float]]:
    """Turn a 2-point straight segment into a gently wavy polyline (wind + jitter)."""
    start_x, start_y = start_point
    end_x, end_y = end_point
    delta_x, delta_y = end_x - start_x, end_y - start_y
    distance = math.hypot(delta_x, delta_y)
    if distance < 1e-6:
        return [{"x": start_x, "y": start_y}, {"x": end_x, "y": end_y}]

    tangent_x, tangent_y = delta_x / distance, delta_y / distance
    normal_x, normal_y = -tangent_y, tangent_x

    # Gentle random-walk wobble; amplitude grows then fades to avoid hooky endpoints
    sample_count = random.randint(CURVE_MIN_WAYPOINTS, CURVE_MAX_WAYPOINTS)
    max_lateral = (
        random.uniform(CURVE_LATERAL_FRAC_MIN, CURVE_LATERAL_FRAC_MAX) * distance
    )
    lateral = 0.0
    lateral_step_sigma = max_lateral * 0.06
    along_drift_sigma = distance * CURVE_ALONG_JITTER_FRAC * 0.08

    points: List[Dict[str, float]] = []
    for i in range(sample_count + 1):
        t = i / float(sample_count)
        base_x = start_x + delta_x * t
        base_y = start_y + delta_y * t

        envelope = math.sin(math.pi * t) ** 1.05
        lateral += random.gauss(0.0, lateral_step_sigma) * envelope * 1.25
        lateral = max(-max_lateral, min(max_lateral, lateral))
        along_jitter = random.gauss(0.0, along_drift_sigma) * envelope

        x = base_x + normal_x * lateral + tangent_x * along_jitter
        y = base_y + normal_y * lateral + tangent_y * along_jitter
        x, y = _clamp_point_to_viewport(x, y, viewport_width, viewport_height)
        points.append({"x": x, "y": y})

    # Ensure exact endpoints
    points[0] = {"x": start_x, "y": start_y}
    points[-1] = {"x": end_x, "y": end_y}
    return points


def _soft_lateral_wobble(t: float, phase1: float, phase2: float) -> float:
    """Two-sine low-frequency lateral wobble in [-1,1], parameterized by path t."""
    return 0.62 * math.sin(
        2.0 * math.pi * (WOBBLE_FREQ_1 * t + phase1)
    ) + 0.38 * math.sin(2.0 * math.pi * (WOBBLE_FREQ_2 * t + phase2))


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
    """Simple windmouse path generator with input guards and runaway protection."""
    try:
        sx, sy = float(start[0]), float(start[1])
        tx, ty = float(target[0]), float(target[1])
        wind = float(wind)
        gravity = float(gravity)
        min_step = float(min_step)
        max_step = float(max_step)
        target_area = float(target_area)
        jitter = float(jitter)
    except Exception as exc:
        logging.getLogger(__name__).warning("windmouse input coercion failed: %s", exc)
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
    """Minimal mouse move emit with fast retries; aborts quickly on persistent CDP stalls."""
    get_mouse_recorder(page).log_move(x, y)
    if cdp is not None:
        last_err: Optional[BaseException] = None
        async with get_mouse_lock(page):
            for attempt in range(3):
                try:
                    # Wrap in task + shield to prevent cancellation crashing zendriver (InvalidStateError)
                    task = asyncio.create_task(
                        page.send(
                            cdp.input_.dispatch_mouse_event(
                                type_="mouseMoved", x=float(x), y=float(y)
                            )
                        )
                    )
                    resp = await asyncio.wait_for(
                        asyncio.shield(task),
                        timeout=0.10,
                    )
                    if isinstance(resp, dict) and resp.get("success") is False:
                        raise RuntimeError(
                            "CDP dispatchMouseEvent returned success=False"
                        )
                    return
                except asyncio.TimeoutError:
                    # Timeout: the shielded task continues in background.
                    # Return immediately to maintain trajectory rhythm (drop this frame)
                    # rather than stalling for retries.
                    return
                except Exception as exc:
                    last_err = exc
                await asyncio.sleep(0.02)

        # If all attempts failed, raise to avoid silently stalling the event loop
        raise RuntimeError(
            f"mouseMoved send failed after retries at ({x:.2f}, {y:.2f})"
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

    # Clamp incoming points to viewport
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
    # Optional bridge from last known position to first point
    if current != (pts[0]["x"], pts[0]["y"]):
        segments.append((current, (pts[0]["x"], pts[0]["y"])))
    for i in range(len(pts) - 1):
        segments.append(
            ((pts[i]["x"], pts[i]["y"]), (pts[i + 1]["x"], pts[i + 1]["y"]))
        )

    # Emit cadence (configurable); default to 15ms between steps for responsiveness.
    interval = float(getattr(cfg, "MOVE_INTERVAL_S", 0.015))
    if interval > 0.1:  # guard against misconfig causing multi-second gaps
        logging.getLogger(__name__).warning(
            "MOVE_INTERVAL_S=%s too large; capping to 15ms for responsiveness",
            interval,
        )
        interval = 0.015
    max_move_duration = float(getattr(cfg, "MAX_MOVE_DURATION_S", 1.0))

    for start, end in segments:
        try:
            path_points = windmouse(start, end)
        except Exception:
            logging.getLogger(__name__).warning(
                "windmouse failed for segment %s->%s; falling back to straight line",
                start,
                end,
                exc_info=True,
            )
            path_points = [start, end]
        if not path_points:
            path_points = [start, end]
        # If path is excessively long, decimate to avoid blocking event loop
        if len(path_points) > 1200:
            logging.getLogger(__name__).warning(
                "windmouse produced %s points; decimating to 1200 to avoid stall",
                len(path_points),
            )
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
