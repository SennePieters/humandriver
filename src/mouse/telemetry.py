from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Awaitable
from pathlib import Path as FSPath
import asyncio
import time
import logging

TrajectoryCallback = Optional[Callable[[FSPath], Awaitable[None]]]
_TRAJECTORY_CALLBACK: TrajectoryCallback = None
_MOUSE_SEND_LOCK = asyncio.Lock()


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


def get_mouse_recorder(page) -> TrajectoryRecorder:
    if not hasattr(page, "_humandriver_mouse_recorder"):
        page._humandriver_mouse_recorder = TrajectoryRecorder()
    return page._humandriver_mouse_recorder
