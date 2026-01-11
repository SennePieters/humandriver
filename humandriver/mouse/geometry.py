from __future__ import annotations
from typing import Any, Tuple, Dict, Union, Optional, Sequence
import time
import asyncio
from zendriver import cdp
from ..utils import clamp, random_uniform


class ViewportUnavailable(RuntimeError):
    """Raised when viewport size cannot be determined from CDP."""

    pass


def _unwrap_zendriver_value(possibly_wrapped: Any) -> Any:
    """Normalize zendriver responses into plain dicts or values."""
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


def _clamp_point_to_viewport(
    x: float, y: float, viewport_width: float, viewport_height: float
) -> Tuple[float, float]:
    """Clamp a point (x,y) into [0,viewport_width]×[0,viewport_height]."""
    return clamp(x, 0.0, viewport_width), clamp(y, 0.0, viewport_height)


def _quad_to_bounding_rect(quad: Sequence[float]) -> Dict[str, float]:
    """Convert an 8-number quad to a bounding rect dict."""
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
    """Inset a rectangle by a fraction of its size on all sides."""
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


def _sample_point_in_rect(rect: Dict[str, float]) -> Tuple[float, float]:
    """Uniformly sample a point inside a rect."""
    x = random_uniform(rect["x"], rect["x"] + rect["width"])
    y = random_uniform(rect["y"], rect["y"] + rect["height"])
    return x, y


async def get_viewport(
    page,
    *,
    timeout_seconds: float = 1.5,
    poll_interval_seconds: float = 0.05,
    debug: bool = False,
) -> Tuple[int, int]:
    """Lightweight viewport fetch with fallback."""
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

    async def _get_box_by_object_id(object_id: str) -> Optional[Dict[str, float]]:
        try:
            resp = await page.send(cdp.dom.get_box_model(object_id=object_id))
            bm = _unwrap_zendriver_value(resp)
            model = bm.get("model") or bm
            content = model.get("content") or bm.get("content")
            if not content or len(content) < 8:
                return None
            return _quad_to_bounding_rect(content)
        except Exception:
            return None

    async def _get_box_by_node_id(node_id: int) -> Optional[Dict[str, float]]:
        try:
            resp = await page.send(cdp.dom.get_box_model(node_id=node_id))
            bm = _unwrap_zendriver_value(resp)
            model = bm.get("model") or bm
            content = model.get("content") or bm.get("content")
            if not content or len(content) < 8:
                return None
            return _quad_to_bounding_rect(content)
        except Exception:
            return None

    # --- Case A: raw CSS selector string ---
    if isinstance(target, str):
        sel = target
        start = time.perf_counter()
        while (time.perf_counter() - start) < timeout_seconds:
            # Attempt 1: Runtime.evaluate -> objectId
            try:
                eval_resp = await page.send(
                    cdp.runtime.evaluate(
                        expression=f"document.querySelector({sel!r})",
                        return_by_value=False,
                        await_promise=False,
                    )
                )
                er = _unwrap_zendriver_value(eval_resp)
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

            # Attempt 2: DOM.getDocument + DOM.querySelector -> nodeId
            try:
                doc_resp = await page.send(cdp.dom.get_document())
                doc = _unwrap_zendriver_value(doc_resp)
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
                    qs = _unwrap_zendriver_value(qs_resp)
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

    # --- Case B: element handle ---
    t = _unwrap_zendriver_value(target)
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
