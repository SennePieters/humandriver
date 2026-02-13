from __future__ import annotations
import random
import math
import logging
from typing import Optional

from ..utils import clamp as _clamp, random_uniform as _rand
from .config import kcfg
from .telemetry import get_keyboard_recorder
from .utils import (
    HiResTimer,
    _base_char_delay_for_wpm,
    _drift,
    _lognormal_delay,
    _looks_like_word_boundary,
    _is_punct,
    _is_special,
    _needs_shift,
    _is_printable,
    _force_typo,
)
from .pacer import _Pacer
from .primitives import (
    _press_tab,
    _press_enter,
    _emit_insert_text,
    _press_backspace,
    _emit_key_down_up,
)
from .analysis import summarize_typing

# Route debug prints in this module through logging
print = logging.getLogger(__name__).debug


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

    rec = get_keyboard_recorder(page)

    # If no text, still honor tab/enter (both if requested)
    if not text:
        if tab:
            await _press_tab(page)
            rec.log("keyDown", "Tab", 0.0)
            rec.log("keyUp", "Tab", 0.0)
        if enter:
            await _press_enter(page)
            rec.log("keyDown", "Enter", 0.0)
            rec.log("keyUp", "Enter", 0.0)
        if log_summary:
            print(summarize_typing(page=page))
        return

    rec.reset(seed=seed)

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
    pacer = _Pacer(w_lo, w_hi, rec)

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
                rec.log("char", wrong, 0.0)
                pacer.on_char()  # count the wrong printable char

                # === Human-like correction sequence ===
                consecutive_errors += 1
                rec.error_count += 1

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
                rec.log("keyDown", "Backspace", 0.0)
                rec.log("keyUp", "Backspace", 0.0)
                pacer.on_edit_key()

                # Brief pause after correction (we will type a char after this)
                await pacer.sleep(
                    _rand(*kcfg.AFTER_CORRECTION_PAUSE), True, "<after-correction>"
                )

                # Insert the correct character using insertText (most reliable)
                await _emit_insert_text(page, ch)
                rec.log("char", ch, 0.0)
                pacer.on_char()

                # End of correction: reset streak
                consecutive_errors = 0

            else:
                # No error: emit the intended char/control
                if ch == "\n":
                    await _press_enter(page)
                    rec.log("keyDown", "Enter", 0.0)
                    rec.log("keyUp", "Enter", 0.0)
                    pacer.on_edit_key()
                    consecutive_errors = 0
                elif _is_printable(ch):
                    await _emit_insert_text(page, ch)
                    rec.log("char", ch, 0.0)
                    pacer.on_char()
                    consecutive_errors = 0
                else:
                    await _emit_key_down_up(page, key=ch, text=ch)
                    rec.log("keyDown", ch, 0.0)
                    rec.log("keyUp", ch, 0.0)
                    pacer.on_edit_key()
                    consecutive_errors = 0

            remaining_in_burst -= 1
            prev_ch = ch

        # Final settle (not followed by char)
        await pacer.sleep(_rand(0.010, 0.035), False, "<settle>")

        # Post-action (Tab/Enter) — honor both if requested
        if tab:
            await _press_tab(page)
            rec.log("keyDown", "Tab", 0.0)
            rec.log("keyUp", "Tab", 0.0)
            pacer.on_edit_key()
        if enter:
            await _press_enter(page)
            rec.log("keyDown", "Enter", 0.0)
            rec.log("keyUp", "Enter", 0.0)
            pacer.on_edit_key()

    if log_summary:
        print(summarize_typing(page=page))
