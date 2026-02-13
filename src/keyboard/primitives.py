from __future__ import annotations
import asyncio
import logging
from typing import Callable, Awaitable, Any, Optional, Dict
from zendriver import cdp
from .config import kcfg


async def _send_cdp_event(
    page, fn: Callable[[], Awaitable[Any]], *, label: str
) -> None:
    """Send a CDP event with a short timeout; fall back to background dispatch."""
    # Create the task once to ensure it runs to completion regardless of timeout
    task = asyncio.create_task(fn())
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=kcfg.CDP_SEND_TIMEOUT_S)
    except asyncio.TimeoutError:
        logging.getLogger(__name__).warning(
            "CDP %s stalled >%.0f ms; continuing in background",
            label,
            kcfg.CDP_SEND_TIMEOUT_S * 1000.0,
        )
        # Task continues in background; no need to re-schedule
    except Exception:
        logging.getLogger(__name__).warning(
            "CDP %s failed (skipped this event)", label, exc_info=True
        )


async def _emit_insert_text(page, text: str) -> None:
    await _send_cdp_event(
        page,
        lambda: page.send(cdp.input_.insert_text(text=text)),
        label="insertText",
    )


async def _emit_key_down_up(
    page, key: str, code: Optional[str] = None, text: Optional[str] = None
) -> None:
    kwargs: Dict[str, Any] = {"key": key}
    if code:
        kwargs["code"] = code
    if text is not None:
        kwargs["text"] = text
    await _send_cdp_event(
        page,
        lambda: page.send(cdp.input_.dispatch_key_event(type_="keyDown", **kwargs)),
        label="keyDown",
    )
    # no sleep here; caller will pace
    await _send_cdp_event(
        page,
        lambda: page.send(cdp.input_.dispatch_key_event(type_="keyUp", **kwargs)),
        label="keyUp",
    )


async def _press_backspace(page) -> None:
    await _send_cdp_event(
        page,
        lambda: page.send(
            cdp.input_.dispatch_key_event(
                type_="rawKeyDown",
                key="Backspace",
                code="Backspace",
                windows_virtual_key_code=8,
                native_virtual_key_code=8,
            )
        ),
        label="backspaceDown",
    )
    await _send_cdp_event(
        page,
        lambda: page.send(
            cdp.input_.dispatch_key_event(
                type_="keyUp",
                key="Backspace",
                code="Backspace",
                windows_virtual_key_code=8,
                native_virtual_key_code=8,
            )
        ),
        label="backspaceUp",
    )


async def _press_enter(page) -> None:
    await _send_cdp_event(
        page,
        lambda: page.send(
            cdp.input_.dispatch_key_event(
                type_="keyDown",
                key="Enter",
                code="Enter",
                windows_virtual_key_code=13,
                native_virtual_key_code=13,
            )
        ),
        label="enterDown",
    )
    await _send_cdp_event(
        page,
        lambda: page.send(
            cdp.input_.dispatch_key_event(
                type_="keyUp",
                key="Enter",
                code="Enter",
                windows_virtual_key_code=13,
                native_virtual_key_code=13,
            )
        ),
        label="enterUp",
    )


async def _press_tab(page) -> None:
    await _send_cdp_event(
        page,
        lambda: page.send(
            cdp.input_.dispatch_key_event(
                type_="rawKeyDown",
                key="Tab",
                code="Tab",
                windows_virtual_key_code=9,
                native_virtual_key_code=9,
            )
        ),
        label="tabDown",
    )
    await _send_cdp_event(
        page,
        lambda: page.send(
            cdp.input_.dispatch_key_event(
                type_="keyUp",
                key="Tab",
                code="Tab",
                windows_virtual_key_code=9,
                native_virtual_key_code=9,
            )
        ),
        label="tabUp",
    )
