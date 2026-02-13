from __future__ import annotations
from ..utils import clamp as _clamp
from .config import kcfg
from .utils import _sleep


class _Pacer:
    def __init__(self, wpm_lo: float, wpm_hi: float, recorder):
        self.cps_min = (wpm_lo * 5.0) / 60.0  # min chars/second allowed
        self.cps_max = (wpm_hi * 5.0) / 60.0  # max chars/second allowed
        self.elapsed = 0.0  # total time we've accounted for
        self.printable_chars = 0  # number of printable chars emitted
        self.recorder = recorder

    def on_char(self) -> None:
        self.printable_chars += 1
        # count a small dispatch overhead so pacing includes real-world time
        self.elapsed += kcfg.DISPATCH_OVERHEAD_S

    def on_edit_key(self) -> None:
        # backspace/enter/tab overhead also affects real time
        self.elapsed += kcfg.DISPATCH_OVERHEAD_S

    def clamp_sleep(self, dt: float, will_emit_char_after: bool) -> float:
        if dt <= 0:
            return 0.0

        # Look-ahead: include the next char if about to type one
        chars_next = self.printable_chars + (1 if will_emit_char_after else 0)

        # Bounds so overall rate stays within [cps_min, cps_max]
        min_elapsed_needed = (chars_next / self.cps_max) if self.cps_max > 0 else 0.0
        max_elapsed_allowed = (
            (chars_next / self.cps_min) if self.cps_min > 0 else float("inf")
        )

        dt_lower = max(0.0, min_elapsed_needed - self.elapsed)  # don't go too fast
        dt_upper = max_elapsed_allowed - self.elapsed  # don't go too slow

        if dt_upper < dt_lower:
            dt_adj = dt_lower
        else:
            dt_adj = _clamp(dt, dt_lower, dt_upper)

        dt_adj = max(dt_adj, kcfg.GLOBAL_MIN_INTERVAL_S)
        return max(0.0, dt_adj)

    async def sleep(self, dt: float, will_emit_char_after: bool, tag: str) -> None:
        dt_adj = self.clamp_sleep(dt, will_emit_char_after)
        self.recorder.log("pause", tag, dt_adj)
        await _sleep(dt_adj)
        self.elapsed += dt_adj
