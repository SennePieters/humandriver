from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable, Awaitable
from zendriver import cdp

import logging
import asyncio
import math
import platform
import random
import time
import ctypes

# Route debug prints in this module through logging
print = logging.getLogger(__name__).debug


# =========================================================
# Telemetry
# =========================================================


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


def get_keyboard_recorder(page) -> KeystrokeRecorder:
    if not hasattr(page, "_humandriver_keyboard_recorder"):
        page._humandriver_keyboard_recorder = KeystrokeRecorder()
    return page._humandriver_keyboard_recorder


# =========================================================
# Config
# =========================================================


class kcfg:
    # Target words per minute (5 chars = 1 word)
    WPM_RANGE = (60, 110)
    WPM_DRIFT_FRAC = 0.20  # gentle session drift (+/-), clamped to WPM_RANGE
    JITTER_COEF = 0.55

    # Bursty rhythm
    BURST_AVG = 7
    BURST_VARIANCE = 4
    BURST_MICRO_PAUSE = (0.045, 0.095)

    # Word & punctuation pauses
    SPACE_PAUSE = (0.010, 0.030)
    WORD_PAUSE = (0.060, 0.140)
    PUNCT_PAUSE = {
        ".": (0.130, 0.240),
        ",": (0.080, 0.160),
        ";": (0.080, 0.160),
        ":": (0.080, 0.160),
        "!": (0.130, 0.240),
        "?": (0.130, 0.240),
        ")": (0.050, 0.120),
    }

    # Extra hesitation for symbols like @#$/\[]{}|~^&*_-+=`"'<> etc.
    SPECIAL_CHAR_PAUSE = (0.450, 0.800)  # ↑ slower than before
    SHIFT_SYMBOL_PAUSE = (0.120, 0.220)  # holding/reaching for SHIFT

    # “Thinking” pauses
    THINK_PAUSE_PROB = 0.12
    THINK_PAUSE = (0.35, 0.75)

    # Error/typo model
    ERROR_RATE = 0.025
    MAX_CONSEC_ERRORS = 3

    # Human-like correction cadence (longer contemplation before fixing)
    CORRECTION_THINK_PAUSE = (0.350, 0.800)  # ↑ more deliberation
    BACKSPACE_PAUSE = (0.120, 0.250)  # ↑ motor pause before backspace
    AFTER_CORRECTION_PAUSE = (0.150, 0.300)  # ↑ settle after fix

    # Low-level timing
    GLOBAL_MIN_INTERVAL_S = 0.009
    TIMING_JITTER_S = 0.004
    KEY_DOWN_UP_GAP = (0.018, 0.040)
    USE_SHIFT_FOR_UPPER_PROB = 0.15

    # Small overhead to account for CDP dispatch etc., per printable char or edit key
    DISPATCH_OVERHEAD_S = 0.0035


CDP_SEND_TIMEOUT_S: float = 0.35  # keep input sends from blocking the loop


async def _send_cdp_event(
    page, fn: Callable[[], Awaitable[Any]], *, label: str
) -> None:
    """Send a CDP event with a short timeout; fall back to background dispatch."""
    # Create the task once to ensure it runs to completion regardless of timeout
    task = asyncio.create_task(fn())
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=CDP_SEND_TIMEOUT_S)
    except asyncio.TimeoutError:
        logging.getLogger(__name__).warning(
            "CDP %s stalled >%.0f ms; continuing in background",
            label,
            CDP_SEND_TIMEOUT_S * 1000.0,
        )
        # Task continues in background; no need to re-schedule
    except Exception:
        logging.getLogger(__name__).warning(
            "CDP %s failed (skipped this event)", label, exc_info=True
        )


# =========================================================
# Hi-res timer
# =========================================================


class HiResTimer:
    def __enter__(self):
        if ctypes and platform.system() == "Windows":
            ctypes.windll.winmm.timeBeginPeriod(1)
        return self

    def __exit__(self, exc_type, exc, tb):
        if ctypes and platform.system() == "Windows":
            ctypes.windll.winmm.timeEndPeriod(1)


# =========================================================
# Helpers
# =========================================================

_PRINTABLE_EXCEPTIONS = set("\n\t\r")


def _is_printable(ch: str) -> bool:
    if not ch or ch in _PRINTABLE_EXCEPTIONS:
        return False
    return 32 <= ord(ch) <= 0x10FFFF


def _sleep(dt: float) -> asyncio.Future:
    return asyncio.sleep(max(0.0, dt))


def _rand(a: float, b: float) -> float:
    lo, hi = (a, b) if a <= b else (b, a)
    return random.uniform(lo, hi)


def _lognormal_delay(base_dt: float, jitter: float) -> float:
    z = random.gauss(0.0, 1.0) * jitter
    return base_dt * math.exp(0.35 * z)


def _chars_per_second_for_wpm(wpm: float) -> float:
    return (wpm * 5.0) / 60.0


def _base_char_delay_for_wpm(wpm: float) -> float:
    cps = max(1e-3, _chars_per_second_for_wpm(wpm))
    return 1.0 / cps


def _drift(value: float, frac: float) -> float:
    return value * (1.0 + random.uniform(-frac, frac))


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _looks_like_word_boundary(prev_ch: str, ch: str) -> bool:
    return (prev_ch.isalnum()) and (ch == " ")


def _is_punct(ch: str) -> bool:
    return ch in kcfg.PUNCT_PAUSE


def _is_special(ch: str) -> bool:
    """Printable non-alnum, non-space, and not in our punct map."""
    return _is_printable(ch) and (not ch.isalnum()) and ch != " " and not _is_punct(ch)


# crude detection of characters that typically require SHIFT on a US layout
_SHIFT_NEEDED = set("""~!@#$%^&*()_+{}|:"<>?""")


def _needs_shift(ch: str) -> bool:
    return (ch in _SHIFT_NEEDED) or (ch.isalpha() and ch.isupper())


# =========================================================
# Simple QWERTY adjacency for plausible slips
# =========================================================

_KEY_NEIGHBORS = {
    "a": "qwsz",
    "s": "awedxz",
    "d": "serfcx",
    "f": "drtgcv",
    "g": "ftyhbv",
    "h": "gyujnb",
    "j": "huikmn",
    "k": "jiolm,",
    "l": "kop;.,",
    "e": "wsd34r",
    "r": "etf45t",
    "t": "ryg56y",
    "y": "tuh67u",
    "u": "yij78i",
    "i": "uok89o",
    "o": "ip[l0p",
    "n": "bhjm",
    "m": "njk,",
    " ": " ",
}


def _force_typo(ch: str) -> str:
    """
    Return a printable typo that is GUARANTEED to differ from ch.
    Prefers keyboard neighbors; falls back to another letter.
    """
    if not _is_printable(ch) or len(ch) != 1:
        return ch  # don't force typos for non-printables

    base = ch.lower()
    pool = list(_KEY_NEIGHBORS.get(base, ""))
    candidates = [c for c in pool if c.lower() != base]

    if not candidates:
        alphabet = "etaoinshrdlcumwfgypbvkjxqz"
        candidates = [c for c in alphabet if c != base]

    t = random.choice(candidates) if candidates else ("x" if base != "x" else "z")
    return t.upper() if ch.isupper() else t


# =========================================================
# Pacer: guarantees overall WPM stays within WPM_RANGE
# =========================================================


class _Pacer:
    def __init__(self, wpm_lo: float, wpm_hi: float, recorder: KeystrokeRecorder):
        self.recorder = recorder
        self.cps_min = (wpm_lo * 5.0) / 60.0  # min chars/second allowed
        self.cps_max = (wpm_hi * 5.0) / 60.0  # max chars/second allowed
        self.elapsed = 0.0  # total time we've accounted for
        self.printable_chars = 0  # number of printable chars emitted

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


# =========================================================
# CDP primitives
#   NOTE: no internal sleeps here — all timing goes through the pacer
# =========================================================


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


async def _hold_shift_type_char(page, ch: str) -> None:
    # Kept for completeness; normal text path uses insertText.
    base = ch.lower()
    await _send_cdp_event(
        page,
        lambda: page.send(
            cdp.input_.dispatch_key_event(
                type_="keyDown", key="Shift", code="ShiftLeft"
            )
        ),
        label="shiftDown",
    )
    await _send_cdp_event(
        page,
        lambda: page.send(
            cdp.input_.dispatch_key_event(type_="keyDown", key=base.upper(), text=ch)
        ),
        label="shiftedCharDown",
    )
    await _send_cdp_event(
        page,
        lambda: page.send(
            cdp.input_.dispatch_key_event(type_="keyUp", key=base.upper(), text=ch)
        ),
        label="shiftedCharUp",
    )
    await _send_cdp_event(
        page,
        lambda: page.send(
            cdp.input_.dispatch_key_event(type_="keyUp", key="Shift", code="ShiftLeft")
        ),
        label="shiftUp",
    )


# =========================================================
# Typing function
# =========================================================


async def type_in_element(
    page,
    text: str = "",  # <-- NOW OPTIONAL
    *,
    wpm: Optional[float] = None,
    accuracy: float = 1.0 - kcfg.ERROR_RATE,
    think_at_start: bool = False,
    seed: Optional[int] = None,
    tab: bool = False,
    enter: bool = False,
    log_summary: bool = False,
) -> None:
    """
    Type text into the already-focused element like a human.
    Use your mouse to focus first.

    - If `text` is empty, you can use this to just send Tab and/or Enter.
      Both flags are honored (Tab first, then Enter).
    """
    if seed is not None:
        random.seed(seed)

    recorder = get_keyboard_recorder(page)

    # If no text, still honor tab/enter (both if requested)
    if not text:
        if tab:
            await _press_tab(page)
            recorder.log("keyDown", "Tab", 0.0)
            recorder.log("keyUp", "Tab", 0.0)
        if enter:
            await _press_enter(page)
            recorder.log("keyDown", "Enter", 0.0)
            recorder.log("keyUp", "Enter", 0.0)
        if log_summary:
            print(summarize_typing(page=page))
        return

    recorder.reset(seed=seed)

    # --- Speed model (clamped to WPM_RANGE) ---
    w_lo, w_hi = (
        kcfg.WPM_RANGE
        if kcfg.WPM_RANGE[0] <= kcfg.WPM_RANGE[1]
        else (kcfg.WPM_RANGE[1], kcfg.WPM_RANGE[0])
    )
    wpm0 = wpm if wpm is not None else random.uniform(w_lo, w_hi)
    wpm0 = _clamp(_drift(wpm0, kcfg.WPM_DRIFT_FRAC), w_lo, w_hi)
    base_dt = _base_char_delay_for_wpm(wpm0)

    # Pacer to enforce overall WPM bounds
    pacer = _Pacer(w_lo, w_hi, recorder)

    # Burst state
    remaining_in_burst = int(
        max(1, random.gauss(kcfg.BURST_AVG, math.sqrt(kcfg.BURST_VARIANCE)))
    )
    prev_ch = ""
    consecutive_errors = 0

    with HiResTimer():
        # Optional up-front thinking pause
        if think_at_start and random.random() < kcfg.THINK_PAUSE_PROB:
            await pacer.sleep(_rand(*kcfg.THINK_PAUSE), True, "<think-start>")

        for ch in text:
            # Base per-char delay with jitter (maps to configured WPM)
            dt_char = _lognormal_delay(base_dt, kcfg.JITTER_COEF)

            # Natural pauses layered on top
            if _looks_like_word_boundary(prev_ch, ch):
                dt_char += _rand(*kcfg.WORD_PAUSE)
            elif ch == " ":
                dt_char += _rand(*kcfg.SPACE_PAUSE)
            elif _is_punct(ch):
                dt_char += _rand(*kcfg.PUNCT_PAUSE[ch])
            elif _is_special(ch):
                # Special symbols: extra hesitation + SHIFT reach if needed
                dt_char += _rand(*kcfg.SPECIAL_CHAR_PAUSE)
                if _needs_shift(ch):
                    dt_char += _rand(*kcfg.SHIFT_SYMBOL_PAUSE)

            # Bursty micro-pause
            if remaining_in_burst <= 0:
                await pacer.sleep(_rand(*kcfg.BURST_MICRO_PAUSE), True, "<burst>")
                remaining_in_burst = int(
                    max(1, random.gauss(kcfg.BURST_AVG, math.sqrt(kcfg.BURST_VARIANCE)))
                )

            # Emit delay before key (this *will* emit a char or control next)
            await pacer.sleep(dt_char, True, "<char-delay>")

            # Decide whether to make a typo (printable chars only)
            will_err = (
                (random.random() > accuracy)
                and _is_printable(ch)
                and (consecutive_errors < kcfg.MAX_CONSEC_ERRORS)
            )

            if will_err:
                wrong = _force_typo(ch)  # guaranteed different
                await _emit_insert_text(page, wrong)
                recorder.log("char", wrong, 0.0)
                pacer.on_char()  # count the wrong printable char

                # === Human-like correction sequence ===
                consecutive_errors += 1
                recorder.error_count += 1

                # "Oops/think" hesitation before reaching for Backspace
                await pacer.sleep(
                    _rand(*kcfg.CORRECTION_THINK_PAUSE), False, "<correction-think>"
                )

                # Small motor pause before pressing Backspace
                await pacer.sleep(
                    _rand(*kcfg.BACKSPACE_PAUSE), False, "<pre-backspace>"
                )

                # Backspace the wrong char
                await _press_backspace(page)
                recorder.log("keyDown", "Backspace", 0.0)
                recorder.log("keyUp", "Backspace", 0.0)
                pacer.on_edit_key()

                # Brief pause after correction (we will type a char after this)
                await pacer.sleep(
                    _rand(*kcfg.AFTER_CORRECTION_PAUSE), True, "<after-correction>"
                )

                # Insert the correct character using insertText (most reliable)
                await _emit_insert_text(page, ch)
                recorder.log("char", ch, 0.0)
                pacer.on_char()

                # End of correction: reset streak
                consecutive_errors = 0

            else:
                # No error: emit the intended char/control
                if ch == "\n":
                    await _press_enter(page)
                    recorder.log("keyDown", "Enter", 0.0)
                    recorder.log("keyUp", "Enter", 0.0)
                    pacer.on_edit_key()
                    consecutive_errors = 0
                elif _is_printable(ch):
                    await _emit_insert_text(page, ch)
                    recorder.log("char", ch, 0.0)
                    pacer.on_char()
                    consecutive_errors = 0
                else:
                    await _emit_key_down_up(page, key=ch, text=ch)
                    recorder.log("keyDown", ch, 0.0)
                    recorder.log("keyUp", ch, 0.0)
                    pacer.on_edit_key()
                    consecutive_errors = 0

            remaining_in_burst -= 1
            prev_ch = ch

        # Final settle (not followed by char)
        await pacer.sleep(_rand(0.010, 0.035), False, "<settle>")

        # Post-action (Tab/Enter) — honor both if requested
        if tab:
            await _press_tab(page)
            recorder.log("keyDown", "Tab", 0.0)
            recorder.log("keyUp", "Tab", 0.0)
            pacer.on_edit_key()
        if enter:
            await _press_enter(page)
            recorder.log("keyDown", "Enter", 0.0)
            recorder.log("keyUp", "Enter", 0.0)
            pacer.on_edit_key()

    if log_summary:
        print(summarize_typing(page=page))


# =========================================================
# Analysis helpers
# =========================================================


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
    if page is None:
        return "No page provided for typing summary"
    rec = get_keyboard_recorder(page)
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
        f"  Corrections (errors fixed): {recorder.error_count}\n"
        f"  Random seed: {used_seed if used_seed is not None else 'N/A'}"
    )


# =========================================================
# Async wrappers (optional)
# =========================================================


async def summarize_typing_async(seed: Optional[int] = None, page: Any = None) -> str:
    """Async wrapper for summarize_typing so you can `await` it if you prefer."""
    return summarize_typing(seed, page=page)


async def print_typing_summary(seed: Optional[int] = None, page: Any = None) -> None:
    """Async helper that prints the summary."""
    print(summarize_typing(seed, page=page))
