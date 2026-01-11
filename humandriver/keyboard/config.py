from __future__ import annotations
from typing import Tuple, Dict


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
    PUNCT_PAUSE: Dict[str, Tuple[float, float]] = {
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

    # Timeout for CDP operations (keep input sends from blocking the loop)
    CDP_SEND_TIMEOUT_S = 0.35
