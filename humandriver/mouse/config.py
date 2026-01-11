from __future__ import annotations


class cfg:
    """Human-like tuning (refined for more natural movement)"""

    # --- Global movement durations (used by high-level behaviors) ---
    DEFAULT_MOVE_MS = 950
    DEFAULT_REST_MS = 850

    # --- WindMouse / path generation ---
    GRAVITY = 10.0
    WIND = 4.0
    MIN_STEP = 3.0
    MAX_STEP = 12.0
    TARGET_AREA = 8.0
    JITTER = 1.0
    OVERSHOOT_PX = 5.0
    ADD_HESITATION = False

    # --- Target offset rules ---
    TARGET_SAFE_INSET_FRAC = 0.15
    TARGET_MICRO_JITTER_PX = 0.8
    TARGET_BIAS_CENTER = True

    # -------------------------------------------------------------------
    # Dispatcher (timing/speed)
    # -------------------------------------------------------------------
    AVG_SPEED_RANGE_PX_S = (600, 1200)
    MIN_SPEED_PX_PER_MS = 0.07
    MAX_SPEED_PX_PER_MS = 0.75

    TARGET_HZ = 75  # slightly lower -> smoother, less robotic
    TIMING_JITTER_S = 0.007  # increased temporal randomness
    MIN_SLEEP_S = 0.005
    EASE_POWER = 3.0  # stronger ease-in/out for smoother starts/stops

    COALESCE_THRESHOLD_PX = 0.3  # preserve micro jitter (was 0.8)
    SMOOTH_ITERS = 2  # apply extra smoothing for gentler arcs
    SMOOTH_WEIGHT = 0.22

    # --- Bridge from last known mouse position to start ---
    BRIDGE_THRESHOLD_PX = 6.0
    BRIDGE_STEPS_MINMAX = (12, 22)
    BRIDGE_CURVE_JITTER_FRAC = 0.035

    GLOBAL_MIN_INTERVAL_S = 0.011
    FINAL_SNAP_EPS_PX = 0.8  # softer landing

    # --- Legacy/compat ---
    MIN_STEP_SLEEP_S = 0.004
    JITTER_SLEEP_S = 0.004
    MOVE_INTERVAL_S = 0.015  # emit cadence for simplified dispatcher
    MAX_MOVE_DURATION_S = 1.0  # cap total duration of a single dispatched path
