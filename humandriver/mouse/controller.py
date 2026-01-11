from __future__ import annotations
from .behaviors import move_to_element


class MouseController:
    """Tiny façade for high-level behaviors bound to a specific page/tab."""

    def __init__(self, page):
        self.page = page

    async def move_to_element(self, target, *, click: bool = False, rest: bool = True):
        return await move_to_element(
            self.page, target, click=click, move_to_rest_after=rest
        )
