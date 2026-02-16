from __future__ import annotations
from dataclasses import dataclass, field
from typing import (
    TypedDict,
    List,
    Tuple,
    Dict,
    Optional,
    Sequence,
    Any,
    Union,
    Callable,
    Awaitable,
)
from zendriver import cdp
from PIL import Image, ImageDraw
from pathlib import Path as FSPath

import logging
import ctypes
import asyncio
import bisect
import math
import platform
import random
import time


# ==============================================
# SECTION: TELEMETRY (from telemetry.py)
# ==============================================


TrajectoryCallback = Optional[Callable[[FSPath], Awaitable[None]]]
_TRAJECTORY_CALLBACK: TrajectoryCallback = None


def get_mouse_lock(page) -> asyncio.Lock:
    """Get or create a per-page lock for serializing mouse events."""
    if not hasattr(page, "_humandriver_mouse_lock"):
        page._humandriver_mouse_lock = asyncio.Lock()
    return page._humandriver_mouse_lock


def set_trajectory_callback(cb: TrajectoryCallback) -> None:
    """Register an async callback invoked whenever a trajectory JPEG is saved."""
    global _TRAJECTORY_CALLBACK
    _TRAJECTORY_CALLBACK = cb
    logging.getLogger(__name__).info(
        "Mouse trajectory callback %s", "registered" if cb else "cleared"
    )


@dataclass
class MouseEvent:
    """Immutable record of one mouse-related event captured by the recorder.

    Attributes:
        x (float): X pixel coordinate in the viewport when the event was logged.
        y (float): Y pixel coordinate in the viewport when the event was logged.
        t (float): Seconds since the recorder started (monotonic), used for timing analysis.
        kind (str): Event category, one of: "move", "down", "up", "click".
    """

    x: float
    y: float
    t: float  # seconds since start (monotonic)
    kind: str  # "move"|"down"|"up"|"click"


@dataclass
class TrajectoryRecorder:
    """Collects mouse events during a session for analysis and visualization.

    Responsibilities:
      - Stores a time-ordered list of MouseEvent instances.
      - Provides convenience methods to append typed events.
      - Keeps a stable monotonic reference start time to compute relative timings.
      - Can be reset to start a fresh session.

    Typical flow:
      recorder.reset()
      recorder.log_move(x, y)
      recorder.log_click(x, y)
      ...
      events = recorder.events
    """

    events: List[MouseEvent] = field(default_factory=list)
    start_ts: float = field(default_factory=lambda: time.perf_counter())

    def _now(self) -> float:
        """Return current monotonic time offset from the recorder's start.

        Returns:
            float: Seconds since the last reset() or object creation.
        """
        return time.perf_counter() - self.start_ts

    def log_move(self, x: float, y: float) -> None:
        """Append a 'move' event to the trajectory.

        Args:
            x (float): Current X in pixels.
            y (float): Current Y in pixels.
        """
        self.events.append(MouseEvent(x, y, self._now(), "move"))

    def log_down(self, x: float, y: float) -> None:
        """Append a 'down' (mouse button press) event."""
        self.events.append(MouseEvent(x, y, self._now(), "down"))

    def log_up(self, x: float, y: float) -> None:
        """Append an 'up' (mouse button release) event."""
        self.events.append(MouseEvent(x, y, self._now(), "up"))

    def log_click(self, x: float, y: float) -> None:
        """Append a 'click' (press+release) event marker.

        Note: The dispatcher itself sends CDP events for move/press/release.
        This method just records a semantic "click" point for analytics/plots.
        """
        self.events.append(MouseEvent(x, y, self._now(), "click"))

    def reset(self) -> None:
        """Clear all recorded events and reset the time origin to now."""
        self.events.clear()
        self.start_ts = time.perf_counter()


# Singleton recorder
recorder = TrajectoryRecorder()


# ==============================================
# SECTION: TYPES (from types.py)
# ==============================================

Point = Tuple[float, float]


class XY(TypedDict):
    """TypedDict for a point with explicit x/y keys (e.g., path waypoints)."""

    x: float
    y: float


Path = List[XY]


# ==============================================
# SECTION: CONFIG (from config.py) — as `class cfg`
# ==============================================


class cfg:
    """Human-like tuning (refined for more natural movement)"""

    # --- Global movement durations (used by high-level behaviors) ---
    DEFAULT_MOVE_MS = 950
    DEFAULT_REST_MS = 850

    # --- WindMouse / path generation ---
    GRAVITY = 10.0
    WIND = 4.0
    MIN_STEP = 3.0
    MAX_STEP = 12.0
    TARGET_AREA = 8.0
    JITTER = 1.0
    OVERSHOOT_PX = 5.0
    ADD_HESITATION = False

    # --- Bézier / spline (optional) ---
    BEZIER_STEPS = 160
    BEZIER_JITTER = 0.30
    SPLINE_INTERNAL_POINTS = 3
    SPLINE_SAMPLES_PER_SEGMENT = 32
    SPLINE_LATERAL_NOISE = 0.12
    SPLINE_MICRO_JITTER = 0.22

    # --- Strategy selection ---
    PATH_STRATEGY = "windmouse"
    PATH_MIX_WEIGHTS = {"windmouse": 1.0}

    # --- Target offset rules ---
    TARGET_SAFE_INSET_FRAC = 0.15
    TARGET_MICRO_JITTER_PX = 0.8
    TARGET_BIAS_CENTER = True

    # -------------------------------------------------------------------
    # Dispatcher (timing/speed)
    # -------------------------------------------------------------------
    AVG_SPEED_RANGE_PX_S = (600, 1200)
    MIN_SPEED_PX_PER_MS = 0.07
    MAX_SPEED_PX_PER_MS = 0.75

    TARGET_HZ = 75  # slightly lower -> smoother, less robotic
    TIMING_JITTER_S = 0.007  # increased temporal randomness
    MIN_SLEEP_S = 0.005
    EASE_POWER = 3.0  # stronger ease-in/out for smoother starts/stops

    COALESCE_THRESHOLD_PX = 0.3  # preserve micro jitter (was 0.8)
    SMOOTH_ITERS = 2  # apply extra smoothing for gentler arcs
    SMOOTH_WEIGHT = 0.22

    # --- Bridge from last known mouse position to start ---
    BRIDGE_THRESHOLD_PX = 6.0
    BRIDGE_STEPS_MINMAX = (12, 22)
    BRIDGE_CURVE_JITTER_FRAC = 0.035

    GLOBAL_MIN_INTERVAL_S = 0.011
    FINAL_SNAP_EPS_PX = 0.8  # softer landing

    # --- Curvy S-shape polyline (for 2-point moves) ---
    CURVE_MIN_WAYPOINTS = 18
    CURVE_MAX_WAYPOINTS = 28
    CURVE_LATERAL_FRAC_MIN = 0.22
    CURVE_LATERAL_FRAC_MAX = 0.40
    CURVE_ALONG_JITTER_FRAC = 0.08

    # --- Subtle wobble for hand tremor effect ---
    WOBBLE_LATERAL_FRAC = 0.018
    WOBBLE_FREQ_1 = 0.8
    WOBBLE_FREQ_2 = 2.0
    END_JITTER_FRAC = 0.006

    # --- Legacy/compat ---
    MIN_STEP_SLEEP_S = 0.004
    JITTER_SLEEP_S = 0.004
    MOVE_INTERVAL_S = 0.015  # emit cadence for simplified dispatcher
    MAX_MOVE_DURATION_S = 1.0  # cap total duration of a single dispatched path


# ==============================================
# SECTION: GEOMETRY (from geometry.py)
# ==============================================


class ViewportUnavailable(RuntimeError):
    """Raised when viewport size cannot be determined from CDP.

    You typically see this if the page hasn't loaded enough for
    Page.getLayoutMetrics() to return positive clientWidth/Height.
    """

    pass


def _unwrap_zendriver_value(possibly_wrapped: Any) -> Any:
    """Normalize zendriver responses into plain dicts or values.

    Behavior:
      - If input is a tuple, take its first element (zendriver often returns tuples).
      - If input exposes to_json()/to_dict()/dict(), call it to get a dict.
      - Otherwise, return the input as-is or {} if falsy.
    """
    value = possibly_wrapped
    if isinstance(value, tuple):
        value = value[0] if value else {}
    for method_name in ("to_json", "to_dict", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return method()
            except Exception:
                pass
    return value or {}


def _get_attr_or_key(container: Any, name: str, default=None):
    """Safe accessor that works with dicts or attribute-style objects."""
    if isinstance(container, dict):
        return container.get(name, default)
    if hasattr(container, name):
        try:
            return getattr(container, name)
        except Exception:
            return default
    return default


def _clamp_scalar(value: float, lower_bound: float, upper_bound: float) -> float:
    """Clamp a scalar into [lower_bound, upper_bound]."""
    return max(lower_bound, min(upper_bound, value))


def _quad_to_bounding_rect(quad: Sequence[float]) -> Dict[str, float]:
    """Convert an 8-number quad (x1,y1,x2,y2,x3,y3,x4,y4) to a bounding rect dict.

    Returns:
        dict: {x,y,width,height,cx,cy} in CSS pixels.
    """
    xs = [quad[0], quad[2], quad[4], quad[6]]
    ys = [quad[1], quad[3], quad[5], quad[7]]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    width = max(0.0, x_max - x_min)
    height = max(0.0, y_max - y_min)
    return {
        "x": x_min,
        "y": y_min,
        "width": width,
        "height": height,
        "cx": x_min + width / 2.0,
        "cy": y_min + height / 2.0,
    }


def _inset_rect_fraction(
    rect: Dict[str, float], inset_fraction: float
) -> Dict[str, float]:
    """Inset a rectangle by a fraction of its size on all sides and return a new rect."""
    inset_fraction = max(0.0, min(0.25, float(inset_fraction)))
    inset_x = rect["width"] * inset_fraction
    inset_y = rect["height"] * inset_fraction
    x = rect["x"] + inset_x
    y = rect["y"] + inset_y
    width = max(0.0, rect["width"] - 2 * inset_x)
    height = max(0.0, rect["height"] - 2 * inset_y)
    return {
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "cx": x + width / 2.0,
        "cy": y + height / 2.0,
    }


def _random_uniform_between(a: float, b: float) -> float:
    """Uniform random sample between a and b (order agnostic)."""
    lower, upper = (a, b) if a <= b else (b, a)
    return random.uniform(lower, upper)


def _sample_point_in_rect(rect: Dict[str, float]) -> Tuple[float, float]:
    """Uniformly sample a point inside a rect."""
    x = _random_uniform_between(rect["x"], rect["x"] + rect["width"])
    y = _random_uniform_between(rect["y"], rect["y"] + rect["height"])
    return x, y


async def get_viewport(
    page,
    *,
    timeout_seconds: float = 1.5,
    poll_interval_seconds: float = 0.05,
    debug: bool = False,
) -> Tuple[int, int]:
    """
    Lightweight viewport fetch:
      - Try Page.getLayoutMetrics first.
      - Fallback to window.innerWidth/innerHeight.
      - Short polling window to avoid long stalls.
    """
    if cdp is None:
        raise RuntimeError("zendriver.cdp is required to read the viewport size")

    async def _try_cdp() -> Tuple[int, int]:
        raw = await page.send(cdp.page.get_layout_metrics())
        if isinstance(raw, tuple):
            raw = raw[0] if raw else {}
        layout = raw.get("layoutViewport") if isinstance(raw, dict) else None
        layout = layout or getattr(raw, "layoutViewport", None) or raw

        def _val(obj: Any, name: str) -> int:
            try:
                if isinstance(obj, dict):
                    return int(float(obj.get(name, 0)))
                return int(float(getattr(obj, name, 0)))
            except Exception:
                return 0

        w = _val(layout, "clientWidth") or _val(layout, "width")
        h = _val(layout, "clientHeight") or _val(layout, "height")
        return w, h

    async def _try_runtime() -> Tuple[int, int]:
        try:
            resp = await page.send(
                cdp.runtime.evaluate(
                    expression="({w: window.innerWidth || 0, h: window.innerHeight || 0})",
                    return_by_value=True,
                    await_promise=False,
                )
            )
            if isinstance(resp, tuple):
                resp = resp[0] if resp else {}
            res = resp.get("result") if isinstance(resp, dict) else resp
            val = (
                res.get("value")
                if isinstance(res, dict)
                else getattr(res, "value", None)
            )
            w = int(val.get("w", 0)) if isinstance(val, dict) else 0
            h = int(val.get("h", 0)) if isinstance(val, dict) else 0
            return w, h
        except Exception:
            return 0, 0

    start = time.perf_counter()
    last_error: Optional[BaseException] = None
    while (time.perf_counter() - start) < timeout_seconds:
        try:
            w, h = await _try_cdp()
            if w > 0 and h > 0:
                return w, h
        except BaseException as exc:
            last_error = exc

        w2, h2 = await _try_runtime()
        if w2 > 0 and h2 > 0:
            return w2, h2

        await asyncio.sleep(poll_interval_seconds)

    raise TimeoutError(
        f"Viewport did not become ready within {timeout_seconds:.2f}s"
        + (f" last error: {last_error!r}" if last_error else "")
    )


async def get_element_rect(
    page,
    target: Union[str, object],
    *,
    timeout_seconds: float = 6.0,
    poll_interval_seconds: float = 0.15,
    debug: bool = False,
) -> Dict[str, float]:
    """Return {x,y,width,height,cx,cy} for a CSS selector or an element handle."""
    if cdp is None:
        raise RuntimeError("zendriver.cdp is required for get_element_rect")

    def _unwrap(obj: Any) -> Any:
        if isinstance(obj, tuple):
            obj = obj[0] if obj else {}
        for name in ("to_json", "to_dict", "dict"):
            m = getattr(obj, name, None)
            if callable(m):
                try:
                    return m()
                except Exception:
                    pass
        return obj or {}

    async def _get_box_by_object_id(object_id: str) -> Optional[Dict[str, float]]:
        try:
            resp = await page.send(cdp.dom.get_box_model(object_id=object_id))
            bm = _unwrap(resp)
            model = bm.get("model") or bm
            content = model.get("content") or bm.get("content")
            if not content or len(content) < 8:
                return None
            xs = [content[0], content[2], content[4], content[6]]
            ys = [content[1], content[3], content[5], content[7]]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            w = max(0.0, x_max - x_min)
            h = max(0.0, y_max - y_min)
            return {
                "x": x_min,
                "y": y_min,
                "width": w,
                "height": h,
                "cx": x_min + w / 2.0,
                "cy": y_min + h / 2.0,
            }
        except Exception:
            return None

    async def _get_box_by_node_id(node_id: int) -> Optional[Dict[str, float]]:
        try:
            resp = await page.send(cdp.dom.get_box_model(node_id=node_id))
            bm = _unwrap(resp)
            model = bm.get("model") or bm
            content = model.get("content") or bm.get("content")
            if not content or len(content) < 8:
                return None
            xs = [content[0], content[2], content[4], content[6]]
            ys = [content[1], content[3], content[5], content[7]]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            w = max(0.0, x_max - x_min)
            h = max(0.0, y_max - y_min)
            return {
                "x": x_min,
                "y": y_min,
                "width": w,
                "height": h,
                "cx": x_min + w / 2.0,
                "cy": y_min + h / 2.0,
            }
        except Exception:
            return None

    # --- Case A: raw CSS selector string ---
    if isinstance(target, str):
        sel = target

        # Poll: Runtime.evaluate (document.querySelector) -> objectId
        start = time.perf_counter()
        while (time.perf_counter() - start) < timeout_seconds:
            try:
                eval_resp = await page.send(
                    cdp.runtime.evaluate(
                        expression=f"document.querySelector({sel!r})",
                        return_by_value=False,
                        await_promise=False,
                    )
                )
                er = _unwrap(eval_resp)
                res = er.get("result") or er
                object_id = (
                    res.get("objectId")
                    if isinstance(res, dict)
                    else getattr(res, "objectId", None)
                )
                if object_id:
                    box = await _get_box_by_object_id(object_id)
                    if box:
                        return box
            except Exception:
                pass

            # Fallback attempt: DOM.getDocument + DOM.querySelector -> nodeId
            try:
                doc_resp = await page.send(cdp.dom.get_document())
                doc = _unwrap(doc_resp)
                root = doc.get("root") or doc
                root_node_id = (
                    root.get("nodeId")
                    if isinstance(root, dict)
                    else getattr(root, "nodeId", None)
                )
                if root_node_id:
                    qs_resp = await page.send(
                        cdp.dom.query_selector(node_id=root_node_id, selector=sel)
                    )
                    qs = _unwrap(qs_resp)
                    node_id = (
                        qs.get("nodeId")
                        if isinstance(qs, dict)
                        else getattr(qs, "nodeId", None)
                    )
                    if node_id:
                        box = await _get_box_by_node_id(node_id)
                        if box:
                            return box
            except Exception:
                pass

            await asyncio.sleep(poll_interval_seconds)

        raise ValueError(f"Element not found for selector: {sel!r}")

    # --- Case B: element handle from page.select(...) or similar ---
    # Try objectId first (works across more cases), then nodeId.
    t = _unwrap(target)
    object_id = (
        getattr(target, "object_id", None)
        or getattr(target, "objectId", None)
        or (t.get("objectId") if isinstance(t, dict) else None)
    )
    if object_id:
        box = await _get_box_by_object_id(object_id)
        if box:
            return box

    node_id = (
        getattr(target, "node_id", None)
        or getattr(target, "nodeId", None)
        or (t.get("nodeId") if isinstance(t, dict) else None)
    )
    if node_id:
        box = await _get_box_by_node_id(node_id)
        if box:
            return box

    raise ValueError("Provided element does not expose objectId or nodeId")


async def sample_point_in_element(
    rect: Dict[str, float], *, safe_inset_fraction: float = 0.0
) -> Tuple[float, float]:
    """Pick a random interior point in an element, avoiding edges."""
    inner_rect = _inset_rect_fraction(rect, safe_inset_fraction)
    if inner_rect["width"] <= 0 or inner_rect["height"] <= 0:
        return rect["cx"], rect["cy"]
    return _sample_point_in_rect(inner_rect)


# ==============================================
# SECTION: DISPATCHERS (from dispatchers.py)
# ==============================================

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


def _clamp_point_to_viewport(
    x: float, y: float, viewport_width: float, viewport_height: float
) -> Tuple[float, float]:
    """Clamp a point (x,y) into [0,viewport_width]×[0,viewport_height]."""
    return max(0.0, min(viewport_width, x)), max(0.0, min(viewport_height, y))


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
    recorder.log_move(x, y)
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


# ==============================================
# SECTION: BEHAVIORS (from behaviors.py)
# ==============================================

REST_RADIUS_PX_HARD_MAX: int = 140
REST_RADIUS_FRAC_OF_MIN: float = 0.12  # 12% of min(viewport width, height)
CLICK_DOWN_UP_DELAY_S: Tuple[float, float] = (0.028, 0.065)  # randomized within
DEFAULT_SAFE_INSET_FRAC: float = 0.08


def _clamp_point_scalar(value: float, lower: float, upper: float) -> float:
    """Clamp a scalar value into [lower, upper]."""
    return max(lower, min(upper, value))


def _clamp_point(
    x: float, y: float, viewport_width: float, viewport_height: float
) -> Tuple[float, float]:
    """Clamp a point (x,y) into the viewport rectangle."""
    return _clamp_point_scalar(x, 0.0, viewport_width), _clamp_point_scalar(
        y, 0.0, viewport_height
    )


def _compute_rest_radius(viewport_width: float, viewport_height: float) -> float:
    """Compute a small local 'rest' radius based on viewport size."""
    base = REST_RADIUS_FRAC_OF_MIN * min(viewport_width, viewport_height)
    return min(REST_RADIUS_PX_HARD_MAX, base)


def _resolve_mouse_button(name: str = "left") -> Any:
    """Return a CDP-compatible mouse button object (enum or shim)."""
    if cdp is not None:
        button_enum = getattr(cdp.input_, "MouseButton", None)
    else:
        button_enum = None

    if button_enum is not None:
        for attr in (name, name.upper(), name.capitalize(), "primary"):
            if hasattr(button_enum, attr):
                return getattr(button_enum, attr)
        try:
            return button_enum(name)  # some builds allow constructing from a string
        except Exception:
            pass

    class _ButtonShim:
        def __init__(self, value: str):
            self._value = value

        def to_json(self) -> str:
            return self._value

    return _ButtonShim(name)


async def _emit_click(page, x: float, y: float, *, button_name: str = "left") -> None:
    """Emit a full click (press, short human-like delay, release) at (x,y)."""
    button = _resolve_mouse_button(button_name)

    if cdp is not None:
        await _send_cdp_event(
            page,
            lambda: page.send(
                cdp.input_.dispatch_mouse_event(
                    type_="mousePressed",
                    x=float(x),
                    y=float(y),
                    button=button,
                    click_count=1,
                )
            ),
            label="mousePressed",
        )
        await asyncio.sleep(random.uniform(*CLICK_DOWN_UP_DELAY_S))
        await _send_cdp_event(
            page,
            lambda: page.send(
                cdp.input_.dispatch_mouse_event(
                    type_="mouseReleased",
                    x=float(x),
                    y=float(y),
                    button=button,
                    click_count=1,
                )
            ),
            label="mouseReleased",
        )
    setattr(page, "_mouse_pos", (float(x), float(y)))


async def move_to_element(
    page,
    target: Union[str, object, Dict[str, float]],
    *,
    click: bool = False,
    safe_inset_fraction: float = DEFAULT_SAFE_INSET_FRAC,
    move_to_rest_after: bool = False,
    rest_delay_seconds: float = 0.0,
) -> Tuple[float, float]:
    """Move the cursor to a target element and optionally click, then optionally rest nearby."""
    viewport_width, viewport_height = await get_viewport(page)

    # Resolve element rect
    if isinstance(target, dict) and {"x", "y", "width", "height"} <= set(target.keys()):
        rect = {
            "x": float(target["x"]),
            "y": float(target["y"]),
            "width": float(target["width"]),
            "height": float(target["height"]),
        }
        rect["cx"] = rect["x"] + rect["width"] / 2.0
        rect["cy"] = rect["y"] + rect["height"] / 2.0
    else:
        rect = await get_element_rect(page, target, debug=False)

    # Sample a point *inside* the element with a small safe inset
    target_x, target_y = await sample_point_in_element(
        rect, safe_inset_fraction=safe_inset_fraction
    )

    # Clamp to viewport just in case
    target_x, target_y = _clamp_point(
        target_x, target_y, viewport_width, viewport_height
    )

    # Determine a starting point: last known mouse position or viewport center
    starting_position = getattr(page, "_mouse_pos", None)
    if starting_position is None:
        starting_position = (viewport_width / 2.0, viewport_height / 2.0)
    start_x, start_y = _clamp_point(
        float(starting_position[0]),
        float(starting_position[1]),
        viewport_width,
        viewport_height,
    )

    # Build a minimal path; dispatcher will bridge/smooth and clamp per step
    path = [{"x": start_x, "y": start_y}, {"x": target_x, "y": target_y}]
    await dispatch_mouse_path(page, path, target_width_pixels=rect.get("width", None))

    if click:
        await _emit_click(page, target_x, target_y, button_name="left")

    setattr(page, "_mouse_pos", (target_x, target_y))

    if move_to_rest_after:
        if rest_delay_seconds > 0:
            await asyncio.sleep(rest_delay_seconds)
        await move_to_rest(page)

    return target_x, target_y


async def move_to_rest(page) -> Tuple[float, float]:
    """Perform a short, natural-looking 'rest' movement near the current cursor."""
    viewport_width, viewport_height = await get_viewport(page)

    current_position = getattr(page, "_mouse_pos", None)
    if current_position is None:
        current_position = (viewport_width / 2.0, viewport_height / 2.0)
    current_x, current_y = _clamp_point(
        float(current_position[0]),
        float(current_position[1]),
        viewport_width,
        viewport_height,
    )

    rest_radius = _compute_rest_radius(viewport_width, viewport_height)
    theta = random.uniform(0.0, 2.0 * math.pi)
    offset_x = rest_radius * 0.35 * random.random() * math.cos(theta)
    offset_y = rest_radius * 0.35 * random.random() * math.sin(theta)

    target_x, target_y = _clamp_point(
        current_x + offset_x, current_y + offset_y, viewport_width, viewport_height
    )

    path = [{"x": current_x, "y": current_y}, {"x": target_x, "y": target_y}]
    await dispatch_mouse_path(page, path, target_width_pixels=None)

    setattr(page, "_mouse_pos", (target_x, target_y))
    return target_x, target_y


# ==============================================
# SECTION: ANALYSIS (from analysis.py)
# ==============================================


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


# ==============================================
# SECTION: RENDER (from render.py)
# ==============================================


def _quantile(values, q):
    """Robust quantile (0..1). Returns value at the given fraction."""
    if not values:
        return 0.0
    q = min(1.0, max(0.0, float(q)))
    data = sorted(values)
    idx = q * (len(data) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return data[lo]
    frac = idx - lo
    return data[lo] * (1 - frac) + data[hi] * frac


def _speed_to_rgb(speed, v_min, v_max):
    """
    Map speed to RGB:
      - slow  => blue (0, 120, 255)
      - mid   => green (60, 205, 60)
      - fast  => red  (255, 60, 60)
    Uses two-segment interpolation: blue→green→red.
    """
    if v_max <= v_min:
        t = 0.0
    else:
        t = (speed - v_min) / (v_max - v_min)
    t = max(0.0, min(1.0, t))

    if t <= 0.5:
        # blue -> green
        u = t / 0.5
        r0, g0, b0 = (0, 120, 255)
        r1, g1, b1 = (60, 205, 60)
        r = int(r0 + (r1 - r0) * u)
        g = int(g0 + (g1 - g0) * u)
        b = int(b0 + (b1 - b0) * u)
    else:
        # green -> red
        u = (t - 0.5) / 0.5
        r0, g0, b0 = (60, 205, 60)
        r1, g1, b1 = (255, 60, 60)
        r = int(r0 + (r1 - r0) * u)
        g = int(g0 + (g1 - g0) * u)
        b = int(b0 + (b1 - b0) * u)
    return (r, g, b)


async def save_mouse_trajectory_jpeg(
    page,
    outfile: str = "mouse_trajectory.jpg",
    *,
    background_color: Tuple[int, int, int] = (12, 12, 14),
    path_base_color: Tuple[int, int, int] = (
        64,
        200,
        255,
    ),  # (unused now; kept for API compatibility)
    path_line_width: int = 2,
    click_ring_radius: int = 5,
    canvas_margin: int = 20,
    annotate: bool = True,
) -> str:
    """
    Render the session's mouse path into a JPEG image, coloring each segment by
    instantaneous speed (pixels per millisecond). A legend is drawn on the right.
    Rendering is offloaded to a worker thread to avoid blocking the event loop.
    """
    if Image is None or ImageDraw is None:
        raise RuntimeError("Pillow (PIL) is required for save_mouse_trajectory_jpeg")

    viewport_width, viewport_height = await get_viewport(page)
    events_snapshot = [e for e in recorder.events if e.kind in ("move", "click")]

    def _render() -> str:
        canvas_width = viewport_width + canvas_margin * 2 + 80  # extra room for legend
        canvas_height = viewport_height + canvas_margin * 2

        image = Image.new("RGB", (canvas_width, canvas_height), background_color)
        draw = ImageDraw.Draw(image)

        events = events_snapshot
        if not events:
            if annotate:
                draw.text(
                    (canvas_margin, canvas_margin),
                    "No mouse events recorded",
                    fill=(180, 180, 180),
                )
            image.save(outfile, format="JPEG", quality=92, optimize=True)
            return outfile

        move_events = [e for e in events if e.kind == "move"]
        point_speeds = []  # (x,y,speed_px_ms)
        speeds = []
        for i in range(1, len(move_events)):
            e_prev, e_cur = move_events[i - 1], move_events[i]
            dt_ms = max(1.0, (e_cur.t - e_prev.t) * 1000.0)
            dist_px = math.hypot(e_cur.x - e_prev.x, e_cur.y - e_prev.y)
            speed = dist_px / dt_ms
            speeds.append(speed)
            point_speeds.append((e_cur.x, e_cur.y, speed))

        if not point_speeds:
            if annotate:
                draw.text(
                    (canvas_margin, canvas_margin),
                    "Not enough move data",
                    fill=(180, 180, 180),
                )
            image.save(outfile, format="JPEG", quality=92, optimize=True)
            return outfile

        v_min = MIN_SPEED_PX_PER_MS
        v_max = MAX_SPEED_PX_PER_MS
        if v_max <= v_min:
            v_max = v_min + 1e-6

        def clamp_to_viewport(x, y):
            x = canvas_margin + max(0.0, min(viewport_width - 1.0, x))
            y = canvas_margin + max(0.0, min(viewport_height - 1.0, y))
            return x, y

        point_radius = max(2, path_line_width + 1)
        for x, y, speed in point_speeds:
            px, py = clamp_to_viewport(x, y)
            color = _speed_to_rgb(speed, v_min, v_max)
            draw.ellipse(
                [
                    px - point_radius,
                    py - point_radius,
                    px + point_radius,
                    py + point_radius,
                ],
                fill=color,
                outline=None,
            )

        for ev in events:
            if ev.kind == "click":
                x, y = clamp_to_viewport(ev.x, ev.y)
                outline = (255, 200, 80)
                inner_fill = (255, 255, 255)
                glow_radius = click_ring_radius + 4
                draw.ellipse(
                    [
                        x - glow_radius,
                        y - glow_radius,
                        x + glow_radius,
                        y + glow_radius,
                    ],
                    outline=(255, 140, 40),
                    width=1,
                )
                draw.ellipse(
                    [
                        x - click_ring_radius,
                        y - click_ring_radius,
                        x + click_ring_radius,
                        y + click_ring_radius,
                    ],
                    outline=outline,
                    width=2,
                )
                draw.ellipse(
                    [x - 2, y - 2, x + 2, y + 2],
                    fill=inner_fill,
                    outline=outline,
                )

        legend_left = canvas_margin + viewport_width + 20
        legend_top = canvas_margin
        legend_height = max(80, viewport_height - 40)
        legend_width = 18
        for i in range(legend_height):
            t = i / max(1, legend_height - 1)
            speed_here = v_max - t * (v_max - v_min)
            color = _speed_to_rgb(speed_here, v_min, v_max)
            draw.line(
                [
                    (legend_left, legend_top + i),
                    (legend_left + legend_width, legend_top + i),
                ],
                fill=color,
                width=1,
            )

        draw.rectangle(
            [
                legend_left - 1,
                legend_top - 1,
                legend_left + legend_width + 1,
                legend_top + legend_height + 1,
            ],
            outline=(200, 200, 200),
            width=1,
        )

        label_x = legend_left + legend_width + 6
        draw.text(
            (label_x, legend_top - 2), f"fast\n{v_max:.3f} px/ms", fill=(220, 220, 220)
        )
        draw.text(
            (label_x, legend_top + legend_height - 22),
            f"slow\n{v_min:.3f} px/ms",
            fill=(220, 220, 220),
        )

        if annotate and speeds:
            avg = sum(speeds) / len(speeds)
            p95 = _quantile(speeds, 0.95)
            p50 = _quantile(speeds, 0.50)
            summary = (
                f"Points: {len(point_speeds)} | Speed px/ms min {min(speeds):.3f} | "
                f"p50 {p50:.3f} | p95 {p95:.3f} | max {max(speeds):.3f} | avg {avg:.3f}"
            )
            draw.text(
                (canvas_margin, canvas_height - canvas_margin - 14),
                summary,
                fill=(200, 200, 200),
            )

        image.save(outfile, format="JPEG", quality=92, optimize=True)
        return outfile

    outfile_path = await asyncio.to_thread(_render)

    cb = _TRAJECTORY_CALLBACK
    if cb is not None:
        try:
            asyncio.create_task(cb(FSPath(outfile_path)))
        except Exception:
            logging.getLogger(__name__).debug(
                "Failed to dispatch trajectory callback", exc_info=True
            )
    else:
        logging.getLogger(__name__).debug(
            "Trajectory saved to %s but no trajectory callback is registered",
            outfile_path,
        )

    return outfile_path


# ==============================================
# SECTION: CONTROLLER (from controller.py)
# ==============================================


class MouseController:
    """Tiny façade for high-level behaviors bound to a specific page/tab."""

    def __init__(self, page):
        """Initialize with a zendriver page/tab (kept as self.page)."""
        self.page = page

    async def move_to_element(self, target, *, click: bool = False, rest: bool = True):
        """Convenience wrapper for behaviors.move_to_element()."""
        return await move_to_element(
            self.page, target, click=click, move_to_rest_after=rest
        )


# ==============================================
# SECTION: HI_RES_TIMER (from hi_res_timer.py)
# ==============================================


class HiResTimer:
    """Context manager to request 1ms Windows system timer resolution.

    On Windows this reduces sleep jitter/latency for tighter timing loops.
    On other platforms, it is a no-op.
    """

    def __enter__(self):
        """Enter the context; on Windows, calls timeBeginPeriod(1)."""
        if ctypes and platform.system() == "Windows":
            ctypes.windll.winmm.timeBeginPeriod(1)
        return self

    def __exit__(self, exc_type, exc, tb):
        """Exit the context; on Windows, balances with timeEndPeriod(1)."""
        if ctypes and platform.system() == "Windows":
            ctypes.windll.winmm.timeEndPeriod(1)


# End of single-file build (Readable Edition)
