from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import time


@dataclass(frozen=True)
class KeystrokeEvent:
    t: float
    kind: str  # 'char' | 'keyDown' | 'keyUp' | 'pause'
    value: str  # character or key name, or pause tag
    dt: float  # planned delay before emitting this event (seconds)


@dataclass
class KeystrokeRecorder:
    events: List[KeystrokeEvent] = field(default_factory=list)
    start_ts: float = field(default_factory=lambda: time.perf_counter())
    seed: Optional[int] = None
    error_count: int = 0  # number of typo corrections performed

    def _now(self) -> float:
        return time.perf_counter() - self.start_ts

    def log(self, kind: str, value: str, dt: float) -> None:
        self.events.append(KeystrokeEvent(self._now(), kind, value, dt))

    def reset(self, seed: Optional[int] = None) -> None:
        self.events.clear()
        self.start_ts = time.perf_counter()
        self.seed = seed
        self.error_count = 0


recorder = KeystrokeRecorder()


def get_keyboard_recorder(page) -> KeystrokeRecorder:
    if not hasattr(page, "_humandriver_key_recorder"):
        page._humandriver_key_recorder = KeystrokeRecorder()
    return page._humandriver_key_recorder
