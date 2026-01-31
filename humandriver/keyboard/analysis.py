from __future__ import annotations
import math
import logging
from typing import List, Dict, Optional, Any
from .telemetry import recorder as global_recorder, get_keyboard_recorder
from .utils import _is_printable

# Route debug prints in this module through logging
print = logging.getLogger(__name__).debug


def summarize_typing(seed: Optional[int] = None, page: Any = None) -> str:
    """
    Reports:
      - Total duration
      - Overall Avg WPM (includes pauses/corrections/bursts)
      - Keystroke Avg WPM (just per-character delays; adheres to WPM_RANGE)
      - Per-sec WPM (min/avg/max) from printable chars
      - Printable chars & corrections
      - Seed used
    """
    if page:
        rec = get_keyboard_recorder(page)
    else:
        rec = global_recorder
    evs = rec.events
    if len(evs) < 2:
        return "No typing data"

    # Build deltas and collect printable char timestamps
    deltas: List[float] = []
    char_times: List[float] = []
    char_delay_only: List[
        float
    ] = []  # only the <char-delay> pauses (maps to keystroke pacing)

    for i in range(1, len(evs)):
        dt = evs[i].t - evs[i - 1].t
        if dt >= 0:
            deltas.append(dt)

        if evs[i].kind == "char" and _is_printable(evs[i].value):
            char_times.append(evs[i].t)

        if evs[i].kind == "pause" and evs[i].value == "<char-delay>":
            char_delay_only.append(evs[i].dt)

    start_t = evs[0].t
    end_t = evs[-1].t
    total_time = max(0.0, end_t - start_t)
    if total_time <= 0:
        return "Invalid timing data"

    # Overall average WPM (includes all pauses)
    total_chars = len(char_times)
    overall_cps = (total_chars / total_time) if total_time > 0 else 0.0
    overall_wpm = (overall_cps * 60.0) / 5.0

    # Keystroke-only WPM (should reflect configured WPM range)
    if char_delay_only:
        avg_char_dt = sum(char_delay_only) / len(char_delay_only)
        keystroke_cps = 1.0 / avg_char_dt if avg_char_dt > 0 else 0.0
        keystroke_wpm = (keystroke_cps * 60.0) / 5.0
    else:
        keystroke_wpm = 0.0

    # Per-second WPM
    bins: Dict[int, int] = {}
    for t in char_times:
        sec_index = int(math.floor(t - start_t))
        bins[sec_index] = bins.get(sec_index, 0) + 1

    if bins:
        wpm_per_sec = [((count * 60.0) / 5.0) for _, count in sorted(bins.items())]
        persec_min = min(wpm_per_sec)
        persec_avg = sum(wpm_per_sec) / len(wpm_per_sec)
        persec_max = max(wpm_per_sec)
    else:
        persec_min = persec_avg = persec_max = 0.0

    used_seed = rec.seed if rec.seed is not None else seed

    return (
        "Typing Summary:\n"
        f"  Total duration: {total_time:.2f}s\n"
        f"  Overall Avg WPM (with pauses): {overall_wpm:.2f}\n"
        f"  Keystroke Avg WPM (no pauses): {keystroke_wpm:.2f}\n"
        f"  Per-sec WPM (min/avg/max): {persec_min:.2f} / {persec_avg:.2f} / {persec_max:.2f}\n"
        f"  Printable chars: {total_chars}\n"
        f"  Corrections (errors fixed): {rec.error_count}\n"
        f"  Random seed: {used_seed if used_seed is not None else 'N/A'}"
    )


async def summarize_typing_async(seed: Optional[int] = None, page: Any = None) -> str:
    """Async wrapper for summarize_typing so you can `await` it if you prefer."""
    return summarize_typing(seed, page=page)


async def print_typing_summary(seed: Optional[int] = None, page: Any = None) -> None:
    """Async helper that prints the summary."""
    print(summarize_typing(seed, page=page))
