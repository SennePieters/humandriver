from __future__ import annotations
import asyncio
import math
import random
from typing import Union, Tuple, Dict, Any
from zendriver import cdp

from .geometry import get_viewport, get_element_rect, sample_point_in_element
from .dispatchers import dispatch_mouse_path, _send_cdp_event
from .telemetry import get_mouse_recorder

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
        get_mouse_recorder(page).log_click(x, y)
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
