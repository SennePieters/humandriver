from __future__ import annotations
import asyncio
import math
import random
from .config import kcfg

_PRINTABLE_EXCEPTIONS = set("\n\t\r")


def _is_printable(ch: str) -> bool:
    if not ch or ch in _PRINTABLE_EXCEPTIONS:
        return False
    return 32 <= ord(ch) <= 0x10FFFF


def _sleep(dt: float) -> asyncio.Future:
    return asyncio.sleep(max(0.0, dt))


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
