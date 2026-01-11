from __future__ import annotations
import asyncio
import math
import logging
from typing import Tuple
from pathlib import Path as FSPath
from PIL import Image, ImageDraw

from .config import cfg
from .telemetry import recorder
from . import telemetry
from .geometry import get_viewport


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
    Uses two-segment interpolation: blue->green->red.
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

        v_min = cfg.MIN_SPEED_PX_PER_MS
        v_max = cfg.MAX_SPEED_PX_PER_MS
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

    cb = telemetry._TRAJECTORY_CALLBACK
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
