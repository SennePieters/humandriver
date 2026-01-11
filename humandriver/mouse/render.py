from __future__ import annotations
import asyncio
import math
from typing import Tuple
from pathlib import Path as FSPath
from PIL import Image, ImageDraw

from .config import cfg
from .telemetry import recorder, _TRAJECTORY_CALLBACK
from .geometry import get_viewport


def _speed_to_rgb(speed, v_min, v_max):
    if v_max <= v_min:
        t = 0.0
    else:
        t = (speed - v_min) / (v_max - v_min)
    t = max(0.0, min(1.0, t))
    if t <= 0.5:
        u = t / 0.5
        r = int(0 + (60 - 0) * u)
        g = int(120 + (205 - 120) * u)
        b = int(255 + (60 - 255) * u)
    else:
        u = (t - 0.5) / 0.5
        r = int(60 + (255 - 60) * u)
        g = int(205 + (60 - 205) * u)
        b = int(60 + (60 - 60) * u)
    return (r, g, b)


async def save_mouse_trajectory_jpeg(
    page,
    outfile: str = "mouse_trajectory.jpg",
    *,
    background_color: Tuple[int, int, int] = (12, 12, 14),
    path_base_color: Tuple[int, int, int] = (64, 200, 255),
    path_line_width: int = 2,
    click_ring_radius: int = 5,
    canvas_margin: int = 20,
    annotate: bool = True,
) -> str:
    if Image is None or ImageDraw is None:
        raise RuntimeError("Pillow (PIL) is required for save_mouse_trajectory_jpeg")

    viewport_width, viewport_height = await get_viewport(page)
    events_snapshot = [e for e in recorder.events if e.kind in ("move", "click")]

    def _render() -> str:
        canvas_width = viewport_width + canvas_margin * 2 + 80
        canvas_height = viewport_height + canvas_margin * 2
        image = Image.new("RGB", (canvas_width, canvas_height), background_color)
        draw = ImageDraw.Draw(image)

        if not events_snapshot:
            if annotate:
                draw.text(
                    (canvas_margin, canvas_margin),
                    "No mouse events recorded",
                    fill=(180, 180, 180),
                )
            image.save(outfile, format="JPEG", quality=92, optimize=True)
            return outfile

        move_events = [e for e in events_snapshot if e.kind == "move"]
        point_speeds = []
        speeds = []
        for i in range(1, len(move_events)):
            e_prev, e_cur = move_events[i - 1], move_events[i]
            dt_ms = max(1.0, (e_cur.t - e_prev.t) * 1000.0)
            dist_px = math.hypot(e_cur.x - e_prev.x, e_cur.y - e_prev.y)
            speed = dist_px / dt_ms
            speeds.append(speed)
            point_speeds.append((e_cur.x, e_cur.y, speed))

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

        for ev in events_snapshot:
            if ev.kind == "click":
                x, y = clamp_to_viewport(ev.x, ev.y)
                draw.ellipse(
                    [
                        x - click_ring_radius,
                        y - click_ring_radius,
                        x + click_ring_radius,
                        y + click_ring_radius,
                    ],
                    outline=(255, 200, 80),
                    width=2,
                )

        # Legend drawing omitted for brevity but would go here

        image.save(outfile, format="JPEG", quality=92, optimize=True)
        return outfile

    outfile_path = await asyncio.to_thread(_render)
    cb = _TRAJECTORY_CALLBACK
    if cb is not None:
        asyncio.create_task(cb(FSPath(outfile_path)))
    return outfile_path
