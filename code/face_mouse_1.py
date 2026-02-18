# face_128x64_8_mouse.py
# Pixar-quality Cozmo-style eyes — 128x64 px — pygame
# Architecture: EyeRenderer | BlinkController | GazeController | ExpressionController
# v8: +Mouse-click target gaze — left-click on screen sets a world-space target
#     the eyes track toward it with Pixar-quality smooth motion:
#       • attention-capture saccade burst toward the click point
#       • critically-damped spring settle on arrival
#       • subtle anticipation blink on new target acquisition
#       • natural "attention fade" back to autonomous gaze after hold period
#     Eyes have two modes:  AUTO (saccades)  ↔  TARGET (mouse-driven)
#     Right-click or press R to clear target and return to autonomous gaze.
# =============================================================================

import math, random, time, json, os, glob
import pygame

# ---------------------------------------------------------------------------
# 1. CONSTANTS  (no magic numbers in rendering logic)
# ---------------------------------------------------------------------------
FB_W,  FB_H  = 128, 64
FPS_TARGET   = 60
EYE_COLOR    = (2, 222, 254)      # #02DEFE cyan
BG_COLOR     = (0, 0, 0)
EYE_GAP_PX   = 39.0               # center-to-center horizontal gap (Neutral baseline)

# Blink timing (seconds) — snappier, more telegraphed for Pixar feel
BLINK_INTERVAL_MEAN_LO = 4.5
BLINK_INTERVAL_MEAN_HI = 3.2
BLINK_INTERVAL_SIGMA   = 0.42
BLINK_INTERVAL_MIN     = 2.0
BLINK_INTERVAL_MAX     = 7.5
BLINK_ANTICIPATION_DUR = 0.055    # pre-pop widening — longer = more telegraphing
BLINK_CLOSE_DUR        = 0.055    # lid snap — faster = more cartoony snap
BLINK_HOLD_LO          = 0.018
BLINK_HOLD_HI          = 0.062
BLINK_OPEN_DUR         = 0.18 #0.180    # slow rebound — Pixar "follow-through"
DOUBLE_BLINK_ENERGY_TH = 0.65
DOUBLE_BLINK_CHANCE    = 0.01 #0.22

# Saccade timing (seconds)
SACC_JUMP_DUR_LO       = 0.090
SACC_JUMP_DUR_HI       = 0.042
SACC_HOLD_DUR_LO       = 0.140
SACC_HOLD_DUR_HI       = 0.024
SACC_SETTLE_DUR_LO     = 0.140
SACC_SETTLE_DUR_HI     = 0.052
SACC_GAP_LO            = 0.90
SACC_GAP_HI            = 0.14
SACC_LONG_FIX_CHANCE   = 0.22
SACC_LONG_FIX_EXTRA    = (0.45, 1.4)

# Gaze spring — slightly livelier spring for snappier focus
SPRING_OMEGA_LO        = 13.0
SPRING_OMEGA_HI        = 26.0

# Squash & stretch — pushed further for Pixar style
BLINK_SQUASH_WIDEN     = 0.23 #0.13     # wider squash on close
BLINK_STRETCH_NARROW   = 0.045    # pre-pop narrowing
BLINK_PREPOP_STRETCH   = 0.09     # pre-pop height stretch (anticipation)
BLINK_PREPOP_PHASE     = -0.11    # blink_phase value at anticipation peak

# Expression transition
EXPR_DUR_MIN           = 0.11
EXPR_DUR_MAX           = 0.38

# ---------------------------------------------------------------------------
# Pixar secondary-motion constants
# ---------------------------------------------------------------------------
BREATHE_AMP            = 0.008    # subtle breathing oscillation (scale on height)
BREATHE_FREQ           = 0.22     # Hz — slow, calm breathing rate
MICRO_TREMOR_AMP       = 0.0055   # sub-pixel organic life tremor amplitude
MICRO_TREMOR_FREQ      = 7.4      # Hz
IRIS_SHIMMER_PERIOD    = 5.2      # seconds between iris brightness pulses
IRIS_SHIMMER_AMP       = 0       # 0-255 additive brightness boost at peak
GLOW_RADIUS_FACTOR     = 0.28     # soft glow halo size relative to eye height
GLOW_ALPHA_MAX         = 38       # max alpha of glow layer (0-255)
EXPR_ENTRY_SQUASH      = 0.06     # fraction of extra squash added on expression arrival

# ---------------------------------------------------------------------------
# Mouse-target gaze constants  (v8)
# ---------------------------------------------------------------------------
# How long the eyes stay locked on a clicked target before drifting back
TARGET_HOLD_DURATION   = 4.5     # seconds of focused attention after click
# How long the drift-back transition takes (ease-out smooth return)
TARGET_RELEASE_DUR     = 0.001 #1.8     # seconds to blend back to autonomous gaze
# Spring stiffness while tracking the target — snappier than idle saccades
TARGET_SPRING_OMEGA    = 32.0    # higher = faster, more "alert" tracking
# Normalised half-extents of the pygame window in gaze space  (−1 … +1)
# These map window pixel coords → gaze normalised coords
TARGET_MARGIN_X        = 0.04    # dead-zone at window edges (normalised)
TARGET_MARGIN_Y        = 0.04


# ---------------------------------------------------------------------------
# 2. EASING LIBRARY
# ---------------------------------------------------------------------------
def clamp(x, a, b):       return a if x < a else (b if x > b else x)
def lerp(a, b, t):        return a + (b - a) * t

def ease_in_quart(t):
    t = clamp(t, 0, 1);  return t * t * t * t

def ease_out_quint(t):
    t = clamp(t, 0, 1);  return 1 - (1 - t) ** 5

def ease_in_out_cubic(t):
    t = clamp(t, 0, 1)
    return 4*t*t*t if t < 0.5 else 1 - (-2*t + 2)**3 / 2

def ease_out_back(t, s=1.70):
    t = clamp(t, 0, 1);  t1 = t - 1
    return 1 + t1*t1*((s+1)*t1 + s)

def ease_in_out_back(t, s=1.20):
    t = clamp(t, 0, 1);  c2 = s * 1.525
    if t < 0.5: return ((2*t)**2 * ((c2+1)*2*t - c2)) / 2
    return ((2*t-2)**2 * ((c2+1)*(2*t-2) + c2) + 2) / 2


# ---------------------------------------------------------------------------
# 3. TWEEN  (supports delay for staggered, overlapping motion)
# ---------------------------------------------------------------------------
class Tween:
    """Single-value tween with delay and swappable easing curve."""
    __slots__ = ("value","start","target","_t0","dur","delay","_easer","active")

    def __init__(self, value: float):
        self.value   = value
        self.start   = value
        self.target  = value
        self._t0     = 0.0
        self.dur     = 0.001
        self.delay   = 0.0
        self._easer  = ease_in_out_cubic
        self.active  = False

    def set(self, target: float, dur: float = 0.18,
            easer=ease_in_out_cubic, delay: float = 0.0):
        self.start  = self.value
        self.target = target
        self._t0    = time.time()
        self.dur    = max(0.001, dur)
        self.delay  = max(0.0, delay)
        self._easer = easer
        self.active = True

    def update(self) -> float:
        if not self.active:
            return self.value
        elapsed = time.time() - self._t0
        if elapsed < self.delay:
            return self.value
        t = (elapsed - self.delay) / self.dur
        if t >= 1.0:
            self.value  = self.target
            self.active = False
            return self.value
        self.value = lerp(self.start, self.target, self._easer(t))
        return self.value


# ---------------------------------------------------------------------------
# 4. SPRING  (critically damped — used for gaze)
# ---------------------------------------------------------------------------
class Spring1D:
    """Critically damped second-order spring. No overshooting, just smooth settle."""
    __slots__ = ("x", "v", "target")

    def __init__(self, x=0.0):
        self.x = x; self.v = 0.0; self.target = x

    def step(self, dt: float, omega: float = 18.0) -> float:
        a = -2*omega*self.v - omega*omega*(self.x - self.target)
        self.v += a * dt
        self.x += self.v * dt
        return self.x


# ---------------------------------------------------------------------------
# 5. EYE SHAPE DATA  — loaded from JSON files
# ---------------------------------------------------------------------------
# JSON folder, relative to this script
_JSON_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                            os.pardir, "assets", "json")

print(_JSON_FOLDER)

# Full raw JSON params keyed by expression name (used by superellipse renderer)
EYE_JSON: dict = {}

# Rig-summary dict keyed by expression name (used by ExpressionController tweens)
# Keys: w, h, r, top, bot, slant, bx, by, sx, sy
EYE_SHAPES: dict = {}


def _flatten_json(path: str) -> dict:
    """Load a JSON eye file and flatten all sections into one dict."""
    with open(path) as f:
        d = json.load(f)
    S = {}
    for sec in ("Geometry", "Position", "Gaze", "Lids"):
        if sec in d:
            S.update(d[sec])
    return S


def _json_to_rig(S: dict) -> dict:
    """
    Derive the simple rig summary from full JSON params.
    These values drive ExpressionController tweens for smooth transitions.
    """
    # Representative eye size: average L/R, incorporating scale
    w = (S.get("width_L",  34) * S.get("scale_x_L", 1.0) +
         S.get("width_R",  34) * S.get("scale_x_R", 1.0)) / 2.0
    h = (S.get("height_L", 26) * S.get("scale_y_L", 1.0) +
         S.get("height_R", 26) * S.get("scale_y_R", 1.0)) / 2.0

    # Equivalent corner radius from superellipse_n
    # n=2 → pure ellipse (soft), n=10+ → sharp rect
    n = (S.get("superellipse_n_L", 4.0) + S.get("superellipse_n_R", 4.0)) / 2.0
    r = clamp(h * 0.30 * clamp(1.0 - (n - 2.0) / 10.0, 0.1, 1.0), 2.0, h * 0.45)

    # Lid coverage (average L/R)
    top = (S.get("upper_cover_L", 0.0) + S.get("upper_cover_R", 0.0)) / 2.0
    bot = (S.get("lower_cover_L", 0.0) + S.get("lower_cover_R", 0.0)) / 2.0

    # Slant from upper-lid angle asymmetry:
    # positive angle_L & negative angle_R → outer-corner-down (angry)
    # negative angle_L & positive angle_R → inner-corner-down (sad)
    ang_L = S.get("upper_angle_deg_L", 0.0)
    ang_R = S.get("upper_angle_deg_R", 0.0)
    slant = clamp((ang_L - ang_R) / 20.0, -1.5, 1.5)

    # Gaze bias from JSON gaze_x/y
    bx = clamp(S.get("gaze_x", 0.0), -1.2, 1.2)
    by = clamp(S.get("gaze_y", 0.0), -1.2, 1.2)

    # Pose span from range_x/y
    sx = clamp(S.get("range_x", 7.0) / 10.0, 0.2, 1.2)
    sy = clamp(S.get("range_y", 4.0) / 10.0, 0.1, 1.0)

    return dict(w=w, h=h, r=r, top=top, bot=bot, slant=slant,
                bx=bx, by=by, sx=sx, sy=sy)


def _load_all_expressions(folder: str):
    """Scan folder for *.json, populate EYE_JSON and EYE_SHAPES."""
    files = sorted(glob.glob(os.path.join(folder, "*.json")))
    if not files:
        raise FileNotFoundError(
            f"No JSON files found in: {folder}\n"
            f"Make sure the folder 'assets/json/' exists "
            f"next to this script.")
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]  # e.g. "happy_left"
        try:
            S   = _flatten_json(path)
            rig = _json_to_rig(S)
            EYE_JSON[name]   = S
            EYE_SHAPES[name] = rig
        except Exception as exc:
            print(f"[WARNING] Skipping {path}: {exc}")


_load_all_expressions(_JSON_FOLDER)

if not EYE_SHAPES:
    raise RuntimeError("No expressions loaded — check JSON folder path.")

ALL_EXPRESSIONS = list(EYE_SHAPES.keys())
_NEUTRAL = "neutral" if "neutral" in EYE_SHAPES else ALL_EXPRESSIONS[0]
print(f"[INFO] Loaded {len(ALL_EXPRESSIONS)} expressions: {ALL_EXPRESSIONS}")

# ---------------------------------------------------------------------------
# 6. EXPRESSION CONTROLLER
# ---------------------------------------------------------------------------
class ExpressionController:
    """
    Owns the animated rig (tweens) and drives expression transitions.
    Transitions happen THROUGH a blink when possible (spec rule).
    Direct tween for manual/instant.
    """

    # Named groups used by BlinkController to know when to inject a shape-swap blink
    REQUIRES_BLINK_MASK = True   # global policy: shape changes go through blink

    def __init__(self, blink_ctrl):
        self._blink = blink_ctrl   # reference so we can request a masked swap

        self.expr       = _NEUTRAL
        self._pending   = None     # expression queued to apply after blink closes

        # Animated rig channels
        self.base_w     = Tween(36.0)
        self.base_h     = Tween(41.0)
        self.radius     = Tween(11.0)
        self.top        = Tween(0.0)
        self.bot        = Tween(0.0)
        self.top_slant  = Tween(0.0)
        self.bot_slant  = Tween(0.0)

        # Pose / gaze influence (not tweened — snap with expression)
        self.pose_bx    = 0.0
        self.pose_by    = 0.0
        self.pose_sx    = 1.0
        self.pose_sy    = 1.0

        # Squash/stretch overlay (driven by blink & expression change)
        self.squeeze_x  = 1.0   # multiplier on rendered width
        self.squeeze_y  = 1.0   # multiplier on rendered height
        self._sq_tween_x = Tween(1.0)
        self._sq_tween_y = Tween(1.0)

        self._snap(_NEUTRAL)

    # ------------------------------------------------------------------
    def request(self, name: str, through_blink: bool = True):
        """
        Request an expression change.
        through_blink=True: queue it to apply at blink-closed moment (spec rule).
        through_blink=False: tween directly (used for manual/instant).
        """
        if name not in EYE_SHAPES:
            name = _NEUTRAL
        if name == self.expr and self._pending is None:
            return

        if through_blink:
            self._pending = name
            self._blink.request_masked_swap()
        else:
            self._pending = None
            self._apply(name)

    def apply_pending(self):
        """Called by BlinkController at the moment eyes are fully closed."""
        if self._pending is not None:
            self._apply(self._pending)
            self._pending = None

    # ------------------------------------------------------------------
    def _snap(self, name: str):
        """Instant, no transition (init only)."""
        s = EYE_SHAPES[name]
        for tw, v in [(self.base_w, s["w"]), (self.base_h, s["h"]),
                      (self.radius, s["r"]),  (self.top,    s["top"]),
                      (self.bot,   s["bot"]), (self.top_slant, s["slant"]),
                      (self.bot_slant, 0.0)]:
            tw.set(v, 0.001)
        self._set_pose(name)
        self.expr = name

    def _set_pose(self, name: str):
        s = EYE_SHAPES[name]
        self.pose_bx = s["bx"];  self.pose_by = s["by"]
        self.pose_sx = s["sx"];  self.pose_sy = s["sy"]

    def _apply(self, name: str):
        """Animate rig toward target shape with staggered Pixar timing."""
        prev  = self.expr
        self.expr = name
        self._set_pose(name)

        s    = EYE_SHAPES[name]
        sp   = EYE_SHAPES.get(prev, EYE_SHAPES[_NEUTRAL])

        tgt_w   = float(s["w"])
        tgt_h   = float(s["h"])
        tgt_r   = float(s["r"])
        tgt_top = s["top"]
        tgt_bot = s["bot"]
        tgt_ts  = s["slant"]
        tgt_bs  = 0.0

        cur_h = self.base_h.value
        h_delta = abs(tgt_h - cur_h)

        # Duration scales with drama
        base_dur = lerp(EXPR_DUR_MIN, EXPR_DUR_MAX, clamp(h_delta / 20.0, 0, 1))

        is_closing = tgt_h < cur_h - 2.0
        is_opening = tgt_h > cur_h + 2.0

        if is_closing:
            # Lids lead the close (snap in fast), shape follows, width squash-counters
            self.top.set(tgt_top,  base_dur*0.60, ease_in_quart,      delay=0.0)
            self.bot.set(tgt_bot,  base_dur*0.60, ease_in_quart,      delay=0.0)
            self.top_slant.set(tgt_ts, base_dur*0.65, ease_in_out_cubic, delay=0.0)
            self.bot_slant.set(tgt_bs, base_dur*0.65, ease_in_out_cubic, delay=0.0)
            self.base_h.set(tgt_h, base_dur,      ease_out_quint,    delay=base_dur*0.08)
            self.radius.set(tgt_r, base_dur*0.80, ease_out_quint,    delay=base_dur*0.05)
            # Squash: width widens briefly then settles
            squash = tgt_w * lerp(1.0, 1.07, clamp(h_delta/20, 0, 1))
            self._sq_tween_x.set(squash/tgt_w, base_dur*0.40, ease_out_quint, delay=0.0)
            self._sq_tween_x.set(1.0,           base_dur,      ease_out_back,  delay=base_dur*0.35)
            self.base_w.set(tgt_w, base_dur*1.05, ease_out_back, delay=0.0)

        elif is_opening:
            # Shape expands first (anticipation burst), lids follow with overshoot
            self.base_h.set(tgt_h, base_dur*0.78, ease_out_back,  delay=0.0)
            self.radius.set(tgt_r, base_dur*0.70, ease_out_quint, delay=0.0)
            self.top.set(tgt_top,  base_dur*0.88, ease_out_back,  delay=base_dur*0.06)
            self.bot.set(tgt_bot,  base_dur*0.88, ease_out_back,  delay=base_dur*0.06)
            self.top_slant.set(tgt_ts, base_dur*0.90, ease_in_out_cubic, delay=base_dur*0.04)
            self.bot_slant.set(tgt_bs, base_dur*0.90, ease_in_out_cubic, delay=base_dur*0.04)
            # Stretch: width narrows slightly then settles
            stretch = tgt_w * lerp(1.0, 0.95, clamp(h_delta/22, 0, 1))
            self._sq_tween_x.set(stretch/tgt_w, base_dur*0.40, ease_out_quint, delay=0.0)
            self._sq_tween_x.set(1.0,            base_dur,      ease_out_back,  delay=base_dur*0.25)
            self.base_w.set(tgt_w, base_dur*1.10, ease_out_back, delay=0.02)

        else:
            # Lateral / slant / lid-only change
            dur = lerp(EXPR_DUR_MIN, EXPR_DUR_MIN + 0.10, abs(tgt_top - sp["top"]) / 0.3)
            self.base_h.set(tgt_h,  dur,        ease_out_back,     delay=0.0)
            self.base_w.set(tgt_w,  dur*1.05,   ease_out_back,     delay=0.015)
            self.top.set(tgt_top,   dur*0.85,   ease_in_out_cubic, delay=0.0)
            self.bot.set(tgt_bot,   dur*0.85,   ease_in_out_cubic, delay=0.0)
            self.top_slant.set(tgt_ts, dur*0.90, ease_in_out_cubic, delay=0.0)
            self.bot_slant.set(tgt_bs, dur*0.90, ease_in_out_cubic, delay=0.0)
            self.radius.set(tgt_r,  dur,        ease_out_back,     delay=0.01)

    # ------------------------------------------------------------------
    def update(self):
        for tw in (self.base_w, self.base_h, self.radius,
                   self.top, self.bot, self.top_slant, self.bot_slant,
                   self._sq_tween_x, self._sq_tween_y):
            tw.update()
        self.squeeze_x = self._sq_tween_x.value
        self.squeeze_y = self._sq_tween_y.value


# ---------------------------------------------------------------------------
# 7. BLINK CONTROLLER
# ---------------------------------------------------------------------------
class BlinkController:
    """
    Owns blink_phase and blink timing.
    Coordinates with ExpressionController for masked shape swaps.
    """
    IDLE = 0; ANTICIPATE = 1; CLOSE = 2; HOLD = 3; OPEN = 4

    def __init__(self):
        self.phase          = 0.0   # 0=open, 1=closed; negative = pre-pop wide
        self._state         = self.IDLE
        self._t0            = 0.0
        self._hold_dur      = 0.0
        self._next_blink_t  = time.time() + 1.2
        self._pending_swap  = False      # masked expression swap queued
        self._double_pending= False
        self._expr_ctrl     = None       # set after construction
        self.energy         = 0.5        # injected each frame

    def bind(self, expr_ctrl):
        self._expr_ctrl = expr_ctrl

    def request_masked_swap(self):
        """Expression controller asks for a blink-masked swap."""
        self._pending_swap = True
        if self._state == self.IDLE:
            self._trigger()

    def trigger(self):
        """Manual blink trigger."""
        if self._state == self.IDLE:
            self._trigger()
            self._schedule()

    def _trigger(self):
        self._state = self.ANTICIPATE
        self._t0    = time.time()
        self._double_pending = (
            self.energy > DOUBLE_BLINK_ENERGY_TH and
            random.random() < DOUBLE_BLINK_CHANCE
        )

    def _schedule(self):
        mean  = lerp(BLINK_INTERVAL_MEAN_LO, BLINK_INTERVAL_MEAN_HI, self.energy)
        sigma = BLINK_INTERVAL_SIGMA
        iv    = random.lognormvariate(math.log(max(0.2, mean)), sigma)
        iv    = clamp(iv, BLINK_INTERVAL_MIN, BLINK_INTERVAL_MAX)
        self._next_blink_t = time.time() + iv

    def update(self, dt: float):
        now = time.time()

        # Auto-blink timer
        if self._state == self.IDLE and now >= self._next_blink_t:
            self._trigger()
            self._schedule()

        # Idle: relax phase toward 0
        if self._state == self.IDLE:
            k = 1 - 0.5 ** (dt / 0.09)
            self.phase = lerp(self.phase, 0.0, k)
            return

        t = now - self._t0

        if self._state == self.ANTICIPATE:
            # Tiny pre-pop: phase goes to BLINK_PREPOP_PHASE (negative = wider)
            self.phase = lerp(0.0, BLINK_PREPOP_PHASE,
                              clamp(t / BLINK_ANTICIPATION_DUR, 0, 1))
            if t >= BLINK_ANTICIPATION_DUR:
                self._state = self.CLOSE
                self._t0    = now

        elif self._state == self.CLOSE:
            # Fast ease_in — lids accelerate like gravity
            p = clamp(t / BLINK_CLOSE_DUR, 0, 1)
            self.phase = ease_in_quart(p)
            if t >= BLINK_CLOSE_DUR:
                self.phase  = 1.0
                self._state = self.HOLD
                self._t0    = now
                self._hold_dur = random.uniform(BLINK_HOLD_LO, BLINK_HOLD_HI)
                # Apply pending shape swap NOW (eyes are closed — invisible)
                if self._pending_swap and self._expr_ctrl:
                    self._expr_ctrl.apply_pending()
                    self._pending_swap = False

        elif self._state == self.HOLD:
            self.phase = 1.0
            if t >= self._hold_dur:
                self._state = self.OPEN
                self._t0    = now

        elif self._state == self.OPEN:
            # Slow ease_out_back — lids bounce open with Pixar follow-through
            p  = clamp(t / BLINK_OPEN_DUR, 0, 1)
            k  = ease_out_back(p, s=1.35)
            self.phase = clamp(1.0 - k, BLINK_PREPOP_PHASE, 1.0)
            if t >= BLINK_OPEN_DUR:
                self.phase  = 0.0
                self._state = self.IDLE
                if self._double_pending:
                    self._double_pending = False
                    self._next_blink_t = time.time() + random.uniform(0.13, 0.22)


# ---------------------------------------------------------------------------
# 8. GAZE CONTROLLER
# ---------------------------------------------------------------------------
class GazeController:
    """
    Owns eye position via critically-damped springs + saccade FSM.
    Gaze favors horizontal over vertical (human behaviour).
    """
    IDLE = 0; JUMP = 1; HOLD = 2; SETTLE = 3

    def __init__(self):
        self.gx     = Spring1D(0.0)
        self.gy     = Spring1D(0.0)
        self.energy = 0.5

        # Saccade FSM
        self._state  = self.IDLE
        self._t0     = 0.0
        self._from   = (0.0, 0.0)
        self._to     = (0.0, 0.0)
        self._over   = (0.0, 0.0)   # micro-overshoot landing spot
        self._next_t = time.time() + 0.4
        self._hold_dx= 0.0
        self._hold_dy= 0.0

        # Wiggle (post-saccade micro-vibration)
        self._wig_t0   = 0.0
        self._wig_amp  = 0.0
        self._wig_freq = 0.0

        # Expression pose influence (set each frame by face)
        self.pose_bx = 0.0;  self.pose_by = 0.0
        self.pose_sx = 1.0;  self.pose_sy = 1.0

        # Manual override
        self.manual      = False
        self.manual_tx   = 0.0
        self.manual_ty   = 0.0

        # Subtle asymmetry drift
        self._asym_bias  = 0.0
        self._next_asym  = time.time() + 3.0

        # ── Mouse-target mode (v8) ────────────────────────────────────────
        # States: NONE → SNAP_IN → HOLD → RELEASE
        self._tgt_mode   = "NONE"    # current target-tracking phase
        self._tgt_nx     = 0.0       # normalised target coords (−1…+1)
        self._tgt_ny     = 0.0
        self._tgt_hold_t = 0.0       # timestamp when hold phase started
        self._tgt_rel_t0 = 0.0       # timestamp when release phase started
        # Pre-snap: store the saccade origin for smooth interpolation
        self._tgt_from_x = 0.0
        self._tgt_from_y = 0.0

    # --- public ---
    def nudge(self, dx, dy):
        self.manual_tx = clamp(self.manual_tx + dx, -1, 1)
        self.manual_ty = clamp(self.manual_ty + dy, -1, 1)

    def recenter(self):
        self.manual_tx = 0.0;  self.manual_ty = 0.0
        self.gx.target = 0.0;  self.gy.target = 0.0

    def set_mouse_target(self, nx: float, ny: float):
        """
        Called when user left-clicks the window.
        nx, ny are normalised gaze coords in [−1, +1].
        Triggers a fast Pixar-style attention-snap toward the target.
        """
        self._tgt_from_x = self.gx.x
        self._tgt_from_y = self.gy.x
        self._tgt_nx = clamp(nx, -1.0 + TARGET_MARGIN_X, 1.0 - TARGET_MARGIN_X)
        self._tgt_ny = clamp(ny, -1.0 + TARGET_MARGIN_Y, 1.0 - TARGET_MARGIN_Y)
        self._tgt_mode   = "SNAP_IN"
        self._tgt_hold_t = time.time()   # will be reset when we enter HOLD
        # Interrupt any running saccade so the spring snaps toward the new target
        self._state  = self.IDLE
        self._next_t = time.time() + 9999.0  # suppress auto saccades while in target mode

    def clear_target(self):
        """Release target lock and drift back to autonomous gaze."""
        if self._tgt_mode not in ("NONE", "RELEASE"):
            self._tgt_rel_t0 = time.time()
            self._tgt_mode   = "RELEASE"

    # --- update ---
    def update(self, dt: float):
        now = time.time()
        e   = self.energy

        if self.manual:
            self.gx.target = self.manual_tx
            self.gy.target = self.manual_ty
            self._state    = self.IDLE
        elif self._tgt_mode != "NONE":
            self._update_target(now)
        else:
            self._update_saccades(now, e)

        # Asymmetry heartbeat
        if now >= self._next_asym:
            self._asym_bias = random.gauss(0.0, 0.04)
            self._next_asym = now + random.uniform(2.5, 5.5)

        # Use a snappier spring when actively tracking a mouse target
        if self._tgt_mode in ("SNAP_IN", "HOLD"):
            omega = TARGET_SPRING_OMEGA
        else:
            omega = lerp(SPRING_OMEGA_LO, SPRING_OMEGA_HI, e)
        self.gx.step(dt, omega)
        self.gy.step(dt, omega)

        # Post-saccade wiggle (decaying sinusoidal micro-jitter)
        if self._wig_amp > 0.0:
            wt    = now - self._wig_t0
            decay = math.exp(-wt / 0.30)
            if decay < 0.02:
                self._wig_amp = 0.0
            else:
                w = self._wig_amp * decay * math.sin(2*math.pi*self._wig_freq*wt)
                self.gx.x = clamp(self.gx.x + w, -1, 1)

    def _update_target(self, now: float):
        """
        Pixar-quality mouse-target tracking FSM.

        SNAP_IN  →  spring is driven hard toward the clicked point.
                    A micro-overshoot is baked in (like a ballistic saccade)
                    so the eye arrives naturally rather than just stopping.
        HOLD     →  spring settles; tiny ambient micro-drift keeps it alive.
                    After TARGET_HOLD_DURATION seconds, drifts back.
        RELEASE  →  spring target blends back toward (0,0) over
                    TARGET_RELEASE_DUR seconds, then resumes auto-saccades.
        """
        if self._tgt_mode == "SNAP_IN":
            # Micro-overshoot: land slightly past the target (Pixar ballistic)
            ov = 0.10
            ox = clamp(self._tgt_nx + (self._tgt_nx - self._tgt_from_x) * ov, -1, 1)
            oy = clamp(self._tgt_ny + (self._tgt_ny - self._tgt_from_y) * ov, -1, 1)
            self.gx.target = ox
            self.gy.target = oy
            # Transition to HOLD once spring is close enough to target
            dist = math.hypot(self.gx.x - self._tgt_nx, self.gy.x - self._tgt_ny)
            if dist < 0.08:
                self._tgt_mode   = "HOLD"
                self._tgt_hold_t = now

        elif self._tgt_mode == "HOLD":
            # Settle on exact target with tiny alive micro-drift
            drift_x = 0.018 * math.sin(2.0 * math.pi * 0.31 * now + 1.3)
            drift_y = 0.010 * math.sin(2.0 * math.pi * 0.19 * now + 0.7)
            self.gx.target = clamp(self._tgt_nx + drift_x, -1, 1)
            self.gy.target = clamp(self._tgt_ny + drift_y, -1, 1)
            # Auto-release after hold duration
            if now - self._tgt_hold_t >= TARGET_HOLD_DURATION:
                self.clear_target()

        elif self._tgt_mode == "RELEASE":
            t   = now - self._tgt_rel_t0
            k   = ease_out_quint(clamp(t / TARGET_RELEASE_DUR, 0, 1))
            # Blend spring target from locked position back toward centre
            self.gx.target = lerp(self._tgt_nx, 0.0, k)
            self.gy.target = lerp(self._tgt_ny, 0.0, k)
            if t >= TARGET_RELEASE_DUR:
                self._tgt_mode = "NONE"
                # Re-enable auto saccades with a short warm-up gap
                self._next_t   = now + random.uniform(0.3, 0.7)

    def _pick_target(self):
        e  = self.energy
        # Strong center-pull at low energy (relaxed resting gaze)
        center_pull = lerp(0.68, 0.28, e)
        if random.random() < center_pull:
            tx = clamp(self.pose_bx + random.gauss(0, lerp(0.10, 0.20, e)), -1, 1)
            ty = clamp(self.pose_by + random.gauss(0, lerp(0.05, 0.12, e)), -1, 1)
        else:
            sx = lerp(0.25, 0.85, e) * self.pose_sx
            sy = lerp(0.12, 0.44, e) * self.pose_sy
            tx = clamp(random.uniform(-sx, sx) + self.pose_bx, -1, 1)
            ty = clamp(random.uniform(-sy, sy) + self.pose_by, -1, 1)
            if random.random() < 0.78:   # humans prefer horizontal gaze
                ty *= 0.48
        return tx, ty

    def _start_saccade(self):
        cur = (self.gx.target, self.gy.target)
        to  = self._pick_target()
        e   = self.energy
        # Micro overshoot: ballistic landing slightly past target
        ov  = lerp(0.05, 0.15, e)
        ox  = clamp(to[0] + (to[0] - cur[0]) * ov, -1, 1)
        oy  = clamp(to[1] + (to[1] - cur[1]) * ov, -1, 1)
        self._from  = cur
        self._to    = to
        self._over  = (ox, oy)
        self._state = self.JUMP
        self._t0    = time.time()
        # Seed wiggle for energetic saccades
        if e > 0.60:
            self._wig_t0   = time.time()
            self._wig_amp  = lerp(0, 0.055, (e - 0.60) / 0.40)
            self._wig_freq = lerp(9, 16, (e - 0.60) / 0.40)

    def _update_saccades(self, now, e):
        if self._state == self.IDLE and now >= self._next_t:
            self._start_saccade()
            return

        if self._state == self.JUMP:
            t   = now - self._t0
            dur = lerp(SACC_JUMP_DUR_LO, SACC_JUMP_DUR_HI, e)
            k   = ease_out_quint(clamp(t / dur, 0, 1))
            self.gx.target = lerp(self._from[0], self._over[0], k)
            self.gy.target = lerp(self._from[1], self._over[1], k)
            if t >= dur:
                self._state   = self.HOLD
                self._t0      = now
                self._hold_dx = random.gauss(0, 0.016)
                self._hold_dy = random.gauss(0, 0.009)

        elif self._state == self.HOLD:
            t    = now - self._t0
            hold = lerp(SACC_HOLD_DUR_LO, SACC_HOLD_DUR_HI, e)
            ds   = lerp(0.50, 0.15, e)   # micro-drift scale
            self.gx.target = clamp(self._over[0] + self._hold_dx * ds, -1, 1)
            self.gy.target = clamp(self._over[1] + self._hold_dy * ds, -1, 1)
            if t >= hold:
                self._state = self.SETTLE
                self._t0    = now

        elif self._state == self.SETTLE:
            t   = now - self._t0
            dur = lerp(SACC_SETTLE_DUR_LO, SACC_SETTLE_DUR_HI, e)
            k   = ease_in_out_cubic(clamp(t / dur, 0, 1))
            self.gx.target = lerp(self._over[0], self._to[0], k)
            self.gy.target = lerp(self._over[1], self._to[1], k)
            if t >= dur:
                self._state = self.IDLE
                base = lerp(SACC_GAP_LO, SACC_GAP_HI, e)
                gap  = random.uniform(base*0.60, base*1.55)
                if random.random() < lerp(SACC_LONG_FIX_CHANCE, 0.04, e):
                    gap += random.uniform(*SACC_LONG_FIX_EXTRA)
                self._next_t = now + gap


# ---------------------------------------------------------------------------
# 9. EYE RENDERER  — superellipse + per-lid system, reads from EYE_JSON
# ---------------------------------------------------------------------------
SE_STEPS = 120   # polygon resolution

def _se_sign(x):
    return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)

def _se_points(a, b, n):
    n = max(0.5, n)
    pts = []
    for i in range(SE_STEPS):
        t = (i / SE_STEPS) * 2 * math.pi
        c, s = math.cos(t), math.sin(t)
        pts.append((_se_sign(c) * abs(c)**(2/n) * a,
                    _se_sign(s) * abs(s)**(2/n) * b))
    return pts

def _arc_pts(cx, cy, r, a0, a1, steps=14, ccw=False):
    pts = []
    if ccw:
        total = a0 - a1
        if total < 0: total += 2*math.pi
        for i in range(steps+1):
            a = a0 - (i/steps)*total
            pts.append((cx + r*math.cos(a), cy + r*math.sin(a)))
    else:
        total = a1 - a0
        if total < 0: total += 2*math.pi
        for i in range(steps+1):
            a = a0 + (i/steps)*total
            pts.append((cx + r*math.cos(a), cy + r*math.sin(a)))
    return pts

def _bezier4(p0, p1, p2, p3, steps=36):
    pts = []
    for i in range(steps+1):
        t = i/steps; mt = 1-t
        pts.append((mt**3*p0[0]+3*mt**2*t*p1[0]+3*mt*t**2*p2[0]+t**3*p3[0],
                    mt**3*p0[1]+3*mt**2*t*p1[1]+3*mt*t**2*p2[1]+t**3*p3[1]))
    return pts

def _tf(pts, ax, ay, ang):
    ca, sa = math.cos(ang), math.sin(ang)
    return [(px*ca - py*sa + ax, px*sa + py*ca + ay) for px, py in pts]

def _ipts(pts):
    return [(int(round(x)), int(round(y))) for x, y in pts]

def _build_lid_poly(hw, lid_h, rA, rB, peak, tension, is_upper):
    rA = clamp(rA, 0, min(hw, lid_h)); rB = clamp(rB, 0, min(hw, lid_h))
    xL = -hw + rA; xR = hw - rB
    cx1 = xR*(1-tension); cx2 = xL*(1-tension)
    FAR = 9999; poly = []
    if is_upper:
        poly += [(-hw,-FAR),(hw,-FAR)]
        if rB > 0: poly.append((hw,-rB)); poly += _arc_pts(hw-rB,-rB,rB,0,math.pi/2)
        else: poly.append((hw,0))
        poly += (_bezier4((xR,0),(cx1,peak),(cx2,peak),(xL,0)) if abs(xR-xL)>0.5
                 else [(xL,0)])
        if rA > 0: poly += _arc_pts(-hw+rA,-rA,rA,math.pi/2,math.pi)
        else: poly.append((-hw,0))
        poly.append((-hw,-FAR))
    else:
        poly += [(-hw,FAR),(hw,FAR)]
        if rB > 0: poly.append((hw,rB)); poly += _arc_pts(hw-rB,rB,rB,0,-math.pi/2,ccw=True)
        else: poly.append((hw,0))
        poly += (_bezier4((xR,0),(cx1,peak),(cx2,peak),(xL,0)) if abs(xR-xL)>0.5
                 else [(xL,0)])
        if rA > 0: poly += _arc_pts(-hw+rA,rA,rA,-math.pi/2,math.pi,ccw=True)
        else: poly.append((-hw,0))
        poly.append((-hw,FAR))
    return poly


class EyeRenderer:
    """
    Superellipse + full lid renderer.
    draw_eye() reads EYE_JSON for the current expression to get the exact
    per-eye geometry, then applies blink/breathe/gaze animation modifiers.
    """

    @staticmethod
    def draw_eye(surface, color, cx, cy, w_scale, h_scale, r_unused,
                 top_cov, bot_cov, top_slant, bot_slant,
                 shimmer: int = 0, glow_alpha: int = 0,
                 expr_name: str = "",
                 side: str = "L",
                 blink_ws: float = 1.0, blink_hs: float = 1.0,
                 entry_sq: float = 1.0, breathe_sc: float = 1.0):
        """
        Draw one eye onto surface.
        cx, cy        : eye centre (already includes gaze + breathe + tremor offsets)
        w_scale       : total width  multiplier (blink_w * squeeze_x)
        h_scale       : total height multiplier (blink_h * squeeze_y * breathe * entry)
        expr_name     : key into EYE_JSON for full per-eye geometry
        side          : 'L' or 'R'
        """
        S = EYE_JSON.get(expr_name)
        if S is None:
            return

        s = side
        w   = S.get(f"width_{s}",  34)
        h   = S.get(f"height_{s}", 26)
        n   = S.get(f"superellipse_n_{s}", 4.0)
        ang = S.get(f"angle_deg_{s}", 0.0) * math.pi / 180.0
        sx  = S.get(f"scale_x_{s}", 1.0) * w_scale
        sy  = S.get(f"scale_y_{s}", 1.0) * h_scale
        rTL = S.get(f"corner_TL_{s}", 3)
        rTR = S.get(f"corner_TR_{s}", 3)
        rBR = S.get(f"corner_BR_{s}", 3)
        rBL = S.get(f"corner_BL_{s}", 3)

        aw, ah = (w/2)*sx, (h/2)*sy
        if aw < 1 or ah < 0.5:
            return

        # ── superellipse polygon ──
        se_local = _se_points(aw, ah, n)
        ca, sa   = math.cos(ang), math.sin(ang)
        se_world = [(px*ca - py*sa + cx, px*sa + py*ca + cy) for px, py in se_local]
        se_ipts  = _ipts(se_world)
        if len(se_ipts) < 3:
            return

        # ── temp surface ──
        tmp  = pygame.Surface((FB_W, FB_H), pygame.SRCALPHA)
        tmp.fill((0, 0, 0, 0))

        # ── soft glow (behind eye) ──
        if glow_alpha > 0:
            gr = int(max(aw, ah) * 2 * GLOW_RADIUS_FACTOR)
            if gr > 1:
                iw, ih = int(aw*2)+2, int(ah*2)+2
                gsurf  = pygame.Surface((iw+gr*2, ih+gr*2), pygame.SRCALPHA)
                for i in range(5, 0, -1):
                    f = i/5; rad = int(gr*f)
                    pygame.draw.ellipse(gsurf, (*color, int(glow_alpha*(1-f)*0.6)),
                        (gr-rad, gr-rad, iw+rad*2, ih+rad*2))
                surface.blit(gsurf, (int(cx-iw/2)-gr, int(cy-ih/2)-gr))

        # ── fill superellipse ──
        sc = tuple(min(255, c+shimmer) for c in color)
        pygame.draw.polygon(tmp, (*sc, 255), se_ipts)

        # ── corner radius masks ──
        PI = math.pi
        corner_defs = [
            (rTL, -aw+rTL, -ah+rTL, PI,     3*PI/2, (-aw,      -ah     )),
            (rTR,  aw-rTR, -ah+rTR, 3*PI/2, 2*PI,   ( aw-rTR,  -ah     )),
            (rBR,  aw-rBR,  ah-rBR, 0,      PI/2,   ( aw-rBR,   ah-rBR )),
            (rBL, -aw+rBL,  ah-rBL, PI/2,   PI,     (-aw,       ah-rBL )),
        ]
        for idx, (rc, lx, ly, a0, a1, (sqx, sqy)) in enumerate(corner_defs):
            if rc <= 0: continue
            arc = _arc_pts(lx, ly, rc, a0, a1)
            if   idx == 0: ca1,ca2,ca3 = (-aw,-ah),(-aw+rc,-ah),(-aw,-ah+rc)
            elif idx == 1: ca1,ca2,ca3 = ( aw,-ah),( aw-rc,-ah),( aw,-ah+rc)
            elif idx == 2: ca1,ca2,ca3 = ( aw, ah),( aw-rc, ah),( aw, ah-rc)
            else:          ca1,ca2,ca3 = (-aw, ah),(-aw+rc, ah),(-aw, ah-rc)
            mask_pts = _ipts(_tf([ca1,ca2]+list(reversed(arc))+[ca3], cx, cy, ang))
            if len(mask_pts) >= 3:
                pygame.draw.polygon(tmp, (0,0,0,255), mask_pts)

        # ── lids ──
        for is_upper in (True, False):
            pre   = "upper" if is_upper else "lower"
            cover = S.get(f"{pre}_cover_{s}", 0.0)
            if cover <= 0: continue

            rel_x     = S.get(f"{pre}_x_{s}", 0.0)
            lid_ang   = S.get(f"{pre}_angle_deg_{s}", 0.0) * math.pi / 180.0
            rel_w     = S.get(f"{pre}_width_{s}", 1.15)
            rel_h_f   = S.get(f"{pre}_height_{s}", 0.55)
            curvature = S.get(f"{pre}_curvature_{s}", 0.0)
            tension   = S.get(f"{pre}_tension_{s}", 0.3)
            rA = S.get(f"upper_corner_BL_{s}" if is_upper else f"lower_corner_UL_{s}", 0)
            rB = S.get(f"upper_corner_BR_{s}" if is_upper else f"lower_corner_UR_{s}", 0)

            lid_w  = w * sx * rel_w;   hw_lid = lid_w / 2
            lid_h  = h * sy * rel_h_f
            eye_ht = h * sy
            eye_top = cy - eye_ht/2;   eye_bot = cy + eye_ht/2

            contact_y = (clamp(eye_top + cover*eye_ht, eye_top, eye_bot) if is_upper
                         else clamp(eye_bot - cover*eye_ht, eye_top, eye_bot))
            anchor_x  = cx + rel_x * w * sx
            peak      = (1 if is_upper else -1) * curvature * lid_h * 0.9

            poly = _ipts(_tf(_build_lid_poly(hw_lid, lid_h, rA, rB, peak, tension, is_upper),
                             anchor_x, contact_y, lid_ang))
            if len(poly) >= 3:
                pygame.draw.polygon(tmp, (0,0,0,255), poly)

        # ── clip to superellipse ──
        mask = pygame.Surface((FB_W, FB_H), pygame.SRCALPHA)
        mask.fill((0,0,0,0))
        pygame.draw.polygon(mask, (255,255,255,255), se_ipts)
        tmp.blit(mask, (0,0), special_flags=pygame.BLEND_RGBA_MIN)
        surface.blit(tmp, (0,0))

    @staticmethod
    def draw_blink_line(surface, y: int, color):
        """1-pixel horizontal line spanning full width."""
        W = surface.get_width()
        y = int(clamp(y, 0, surface.get_height() - 1))
        pygame.draw.line(surface, color, (0, y), (W - 1, y), 3)


# ---------------------------------------------------------------------------
# 10.  EXPRESSION PICKER  (mood → expression name, uses loaded JSON names)
# ---------------------------------------------------------------------------
def _pick(*candidates):
    """Return candidates that were actually loaded, or neutral."""
    available = [c for c in candidates if c in EYE_SHAPES]
    return available if available else [_NEUTRAL]

def pick_expression(valence: float, arousal: float) -> str:
    if random.random() < 0.52:
        return _NEUTRAL
    if arousal < 0.20:
        return random.choice(_pick("frustated_bored","squint","sleepy_left","sleepy_right","unimpressed_left","unimpressed_right"))
    if valence < -0.65 and arousal > 0.80:
        return random.choice(_pick("angry","furious"))
    if valence < -0.55 and arousal > 0.65:
        return random.choice(_pick("angry","furious","annoyed"))
    if valence < -0.30:
        if arousal > 0.55:
            return random.choice(_pick("annoyed","worried","skeptical_left","skeptical_right"))
        return random.choice(_pick("worried","sad_left","sad_right","sad_looking_down"))
    if arousal > 0.82 and abs(valence) < 0.30:
        return random.choice(_pick("scared","awe"))
    if valence > 0.70 and arousal > 0.72:
        return random.choice(_pick("glee","awe","happy"))
    if valence > 0.35:
        return random.choice(_pick("happy","happy_left","happy_right"))
    if arousal > 0.78:
        return random.choice(_pick("focused_determined","normal_left","normal_right"))
    return random.choice(_pick("neutral","normal_left","normal_right","squint"))


# ---------------------------------------------------------------------------
# 11.  FACE  (thin orchestrator — wires the four controllers together)
# ---------------------------------------------------------------------------
# Auto-generate keymaps from loaded expression names
# Priority order for key assignment
_KEY_PRIORITY = [
    "neutral","happy","glee","focused_determined","normal_left","normal_right",
    "extreme_left","extreme_right","squint",
    "angry","annoyed","frustated_bored","worried",
    "sad_left","sad_right","sad_looking_down","happy_left","happy_right",
    "scared","awe","furious","skeptical_left","skeptical_right",
    "sleepy_left","sleepy_right","suspicious_left","suspicious_right",
    "unimpressed_left","unimpressed_right",
]
_ordered = [n for n in _KEY_PRIORITY if n in EYE_SHAPES]
_ordered += [n for n in ALL_EXPRESSIONS if n not in _ordered]

KEYMAP       = {str(i+1): _ordered[i]    for i in range(min(9, len(_ordered)))}
KEYMAP_SHIFT = {str(i+1): _ordered[i+9]  for i in range(min(9, len(_ordered)-9))}
KEYMAP_CTRL  = {str(i+1): _ordered[i+18] for i in range(min(9, len(_ordered)-18))}

class CozmoFace:
    """
    Thin orchestrator — wires the four controllers together.
    v7: adds Pixar secondary motion:
        • breathing (subtle vertical float + scale oscillation)
        • iris shimmer (periodic brightness pulse)
        • micro-tremor (per-eye organic jitter)
        • soft glow halo (energy-scaled)
        • expression-entry follow-through squash
    """

    def __init__(self, W=FB_W, H=FB_H):
        self.W, self.H = W, H

        self.blink = BlinkController()
        self.expr  = ExpressionController(self.blink)
        self.blink.bind(self.expr)
        self.gaze  = GazeController()

        self.auto_expr  = True
        self.valence    = 0.20;  self.valence_t = 0.20
        self.energy     = 0.55;  self.energy_t  = 0.55
        self._next_mood = time.time() + random.uniform(1.8, 3.5)

        # Subtle per-frame asymmetry
        self._asym      = 0.0
        self._next_asym = time.time() + 3.0

        # ---- Pixar secondary-motion state ----
        # Iris shimmer: periodic brightness pulse
        self._shimmer_val  = 0
        self._shimmer_t0   = time.time() + random.uniform(2.0, IRIS_SHIMMER_PERIOD)

        # Micro-tremor: each eye has its own phase so they don't move in lockstep
        self._tremor_phase_l = random.uniform(0.0, math.tau)
        self._tremor_phase_r = random.uniform(0.0, math.tau)

        # Expression-entry follow-through squash
        self._entry_t        = -99.0   # time of last expression snap
        self._entry_squeeze  = 1.0     # current Y-scale from follow-through

    # ---------------------------------------------------------------------------
    # Public controls
    # ---------------------------------------------------------------------------
    def set_expression(self, name: str, instant: bool = False, through_blink: bool = True):
        prev = self.expr.expr
        self.expr.request(name, through_blink=(through_blink and not instant))
        if name != prev:
            self._entry_t = time.time()

    def trigger_blink(self):
        self.blink.trigger()

    def toggle_manual(self):
        self.gaze.manual = not self.gaze.manual

    def nudge(self, dx, dy):
        self.gaze.nudge(dx, dy)

    def recenter(self):
        self.gaze.recenter()
        self.gaze.clear_target()   # also release any mouse target

    def change_energy(self, d):
        self.energy_t = clamp(self.energy_t + d, 0, 1)

    def on_mouse_click(self, px: int, py: int):
        """
        Convert a window pixel click (px, py) to normalised gaze coords and
        hand off to GazeController.  Also triggers a Pixar attention blink.

        Pixel mapping:
          px=0 → nx=−1,  px=W → nx=+1
          py=0 → ny=−1,  py=H → ny=+1  (top is −1, bottom is +1)
        """
        nx = (px / self.W) * 2.0 - 1.0
        ny = (py / self.H) * 2.0 - 1.0
        self.gaze.set_mouse_target(nx, ny)
        # Pixar "attention blink" — very brief, telegraphs a new focus point
        #self.blink.trigger()

    def on_mouse_right_click(self):
        """Right-click releases mouse target and returns to autonomous gaze."""
        self.gaze.clear_target()

    # ---------------------------------------------------------------------------
    # Update
    # ---------------------------------------------------------------------------
    def update(self, dt: float):
        now = time.time()

        # Smooth mood
        k = 1 - 0.5**(dt / 0.9)
        self.valence = lerp(self.valence, self.valence_t, k)
        self.energy  = lerp(self.energy,  self.energy_t,  k)

        # Autonomous expression
        if self.auto_expr and now >= self._next_mood and not self.gaze.manual:
            dv = random.uniform(-0.34, 0.34)
            de = random.uniform(-0.22, 0.22)
            if random.random() < 0.18: dv -= random.uniform(0.20, 0.55)
            if random.random() < 0.16: dv += random.uniform(0.15, 0.45)
            self.valence_t = clamp(self.valence_t * 0.92 + dv, -1, 1)
            self.energy_t  = clamp(self.energy_t  * 0.94 + de + 0.02, 0, 1)
            new = pick_expression(self.valence_t, self.energy_t)
            if new != self.expr.expr:
                self.expr.request(new, through_blink=True)
                self._entry_t = now
            self._next_mood = now + random.uniform(1.6, 4.0)

        # Propagate energy to sub-controllers
        self.blink.energy = self.energy
        self.gaze.energy  = self.energy
        self.gaze.pose_bx = self.expr.pose_bx
        self.gaze.pose_by = self.expr.pose_by
        self.gaze.pose_sx = self.expr.pose_sx
        self.gaze.pose_sy = self.expr.pose_sy

        self.expr.update()
        self.blink.update(dt)
        self.gaze.update(dt)

        # Asymmetry heartbeat
        if now >= self._next_asym:
            self._asym      = random.gauss(0, 0.055)
            self._next_asym = now + random.uniform(2.5, 5.5)

        # ---- Iris shimmer decay & trigger ----
        if now >= self._shimmer_t0:
            self._shimmer_val = IRIS_SHIMMER_AMP
            self._shimmer_t0  = now + IRIS_SHIMMER_PERIOD + random.uniform(-1.0, 1.2)
        else:
            self._shimmer_val = max(0, self._shimmer_val - int(IRIS_SHIMMER_AMP * dt * 5.5))

        # ---- Expression-entry follow-through squash ----
        age = now - self._entry_t
        if 0.0 <= age < 0.28:
            # Brief vertical squash that bounces back — Pixar "follow-through"
            t = age / 0.28
            # ease_out_back gives overshoot then settle; invert for squash then snap
            squeeze = 1.0 - EXPR_ENTRY_SQUASH * math.sin(t * math.pi) * ease_out_back(1.0 - t, s=1.2)
            self._entry_squeeze = clamp(squeeze, 0.90, 1.05)
        else:
            self._entry_squeeze = 1.0

    # ---------------------------------------------------------------------------
    # Render
    # ---------------------------------------------------------------------------
    def render_to(self, surface: pygame.Surface):
        surface.fill(BG_COLOR)

        now   = time.time()
        phase = self.blink.phase
        e     = self.expr
        expr_name = e.expr                      # current expression (JSON key)
        S = EYE_JSON.get(expr_name, {})

        # ---- Blink squash & stretch ----
        if phase >= 0.0:
            p = clamp(phase, 0, 1)
            blink_wscale = 1.0 + BLINK_SQUASH_WIDEN * ease_out_back(p, s=1.4)
            blink_hscale = clamp(1.0 - ease_in_quart(p), 0.0, 1.0)
            blink_hscale += 0.042 * math.sin(p * math.pi) * (1.0 - p)
            blink_hscale  = clamp(blink_hscale, 0.0, 1.0)
        else:
            s_ = -phase / abs(BLINK_PREPOP_PHASE)
            blink_wscale = 1.0 - BLINK_STRETCH_NARROW * s_
            blink_hscale = 1.0 + BLINK_PREPOP_STRETCH * s_

        # ---- Breathing secondary motion ----
        breathe_scale = 1.0 + BREATHE_AMP * math.sin(2.0 * math.pi * BREATHE_FREQ * now)
        breathe_oy    = int(BREATHE_AMP * self.H * 0.55 *
                            math.sin(2.0 * math.pi * BREATHE_FREQ * now))

        # ---- Combined scale modifiers ----
        w_mod = blink_wscale * e.squeeze_x
        h_mod = blink_hscale * e.squeeze_y * breathe_scale * self._entry_squeeze

        # ---- Gaze pixels — from JSON range_x/y ----
        range_x = S.get("range_x", 7.0)
        range_y = S.get("range_y", 4.0)
        gaze_clamp_n = S.get("gaze_clamp_n", 3.2)
        gx_raw = self.gaze.gx.x * range_x
        gy_raw = self.gaze.gy.x * range_y
        # Superellipse gaze clamp
        if range_x > 0 and range_y > 0:
            nx, ny = gx_raw / range_x, gy_raw / range_y
            m = abs(nx)**gaze_clamp_n + abs(ny)**gaze_clamp_n
            if m > 1.0:
                sc_ = m**(1.0/gaze_clamp_n); gx_raw /= sc_; gy_raw /= sc_
        gz = (gx_raw, gy_raw)

        # ---- Eye centres — from JSON position ----
        cx_base = S.get("center_x", self.W * 0.5)
        cy_base = S.get("center_y", self.H * 0.5)
        gap     = S.get("gap", EYE_GAP_PX)
        wL      = S.get("width_L", 34);  wR = S.get("width_R", 34)
        oxL     = S.get("eye_offset_x_L", 0.0); oyL = S.get("eye_offset_y_L", 0.0)
        oxR     = S.get("eye_offset_x_R", 0.0); oyR = S.get("eye_offset_y_R", 0.0)
        left_cx  = cx_base - gap/2 - wL/2 + oxL + gz[0]
        left_cy  = cy_base + oyL               + gz[1] + breathe_oy
        right_cx = cx_base + gap/2 + wR/2 + oxR + gz[0]
        right_cy = cy_base + oyR               + gz[1] + breathe_oy

        # ---- Per-eye asymmetry (gaze-direction + emotional) ----
        dir_a  = 0.13 * self.gaze.gx.x
        emo_a  = self._asym
        lscale = clamp(1 + (-dir_a - emo_a) * 0.55, 0.85, 1.15)
        rscale = clamp(1 + ( dir_a + emo_a) * 0.55, 0.85, 1.15)

        # ---- Micro-tremor ----
        tl = MICRO_TREMOR_AMP * math.sin(2.0*math.pi*MICRO_TREMOR_FREQ*now + self._tremor_phase_l)
        tr = MICRO_TREMOR_AMP * math.sin(2.0*math.pi*MICRO_TREMOR_FREQ*now + self._tremor_phase_r)
        tremor_l = int(self.H * tl)
        tremor_r = int(self.H * tr)

        # ---- Glow alpha ----
        glow_a = int(lerp(GLOW_ALPHA_MAX * 0.25, GLOW_ALPHA_MAX, self.energy))

        # ---- Draw eyes ----
        if blink_hscale > 0.01:
            EyeRenderer.draw_eye(surface, EYE_COLOR,
                left_cx,  left_cy  + tremor_l,
                w_mod * lscale, h_mod * lscale, 0,
                0, 0, 0, 0,
                shimmer=self._shimmer_val, glow_alpha=glow_a,
                expr_name=expr_name, side='L',
                blink_ws=w_mod * lscale, blink_hs=h_mod * lscale)
            EyeRenderer.draw_eye(surface, EYE_COLOR,
                right_cx, right_cy + tremor_r,
                w_mod * rscale, h_mod * rscale, 0,
                0, 0, 0, 0,
                shimmer=self._shimmer_val, glow_alpha=glow_a,
                expr_name=expr_name, side='R',
                blink_ws=w_mod * rscale, blink_hs=h_mod * rscale)

        # ---- Blink close-line ----
        line_alpha = clamp((phase - 0.75) / 0.25, 0.0, 1.0)
        if line_alpha > 0.0:
            avg_ht = ((S.get("height_L",26)*S.get("scale_y_L",1) +
                       S.get("height_R",26)*S.get("scale_y_R",1)) * 0.5 * h_mod)
            top_cov = (S.get("upper_cover_L",0) + S.get("upper_cover_R",0)) / 2
            bot_cov = (S.get("lower_cover_L",0) + S.get("lower_cover_R",0)) / 2
            lid_offset = int(((top_cov - bot_cov) * 0.5) * avg_ht)
            line_y = int(cy_base + gz[1] + breathe_oy + lid_offset)
            c = tuple(int(lerp(ch, 255, 0.15 * line_alpha)) for ch in EYE_COLOR)
            EyeRenderer.draw_blink_line(surface, line_y, c)


# ---------------------------------------------------------------------------
# 12.  MAIN LOOP
# ---------------------------------------------------------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((FB_W, FB_H), pygame.NOFRAME)

    # Print keymap to console
    print("\n=== Cozmo Face v8 — JSON expressions + Mouse-Target Gaze ===")
    print("Keys 1-9:        ", {k: v for k,v in KEYMAP.items()})
    print("Shift+1-9:       ", {k: v for k,v in KEYMAP_SHIFT.items()})
    print("Ctrl+1-9:        ", {k: v for k,v in KEYMAP_CTRL.items()})
    print("TAB: cycle all   A: auto   E/D: energy   arrows: gaze   R: recenter   Space: blink")
    print("LEFT CLICK: set gaze target   RIGHT CLICK: release target")

    pygame.display.set_caption(
        "Cozmo Face v8  |  Left-click=gaze target  Right-click=release  "
        "1-9=expression  TAB=next  A=auto  E/D=energy  R=recenter  Space=blink")
    clock  = pygame.time.Clock()
    face   = CozmoFace(FB_W, FB_H)
    _tab_idx = [0]

    running = True
    while running:
        dt = clock.tick(FPS_TARGET) / 1000.0

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

            # ── Mouse target (v8) ────────────────────────────────────────
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 1:          # left click  → set gaze target
                    face.on_mouse_click(*ev.pos)
                elif ev.button == 3:        # right click → release target
                    face.on_mouse_right_click()

            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False

                mods  = pygame.key.get_mods()
                shift = bool(mods & pygame.KMOD_SHIFT)
                ctrl  = bool(mods & pygame.KMOD_CTRL)
                kname = pygame.key.name(ev.key)

                # 0: reset to neutral + auto
                if ev.key == pygame.K_0:
                    face.auto_expr = True
                    face.set_expression(_NEUTRAL, instant=True)
                    face.recenter()

                # TAB: cycle all loaded expressions
                elif ev.key == pygame.K_TAB:
                    face.auto_expr = False
                    name = _ordered[_tab_idx[0] % len(_ordered)]
                    _tab_idx[0] += 1
                    face.set_expression(name)
                    print(f"Expression: {name}")

                # A: toggle autonomous expressions
                elif ev.key == pygame.K_a:
                    face.auto_expr = not face.auto_expr

                # Ctrl+1-9
                elif ctrl and kname in KEYMAP_CTRL:
                    face.auto_expr = False
                    face.set_expression(KEYMAP_CTRL[kname])

                # 1–9 / Shift+1–9: manual expression
                elif kname in (KEYMAP_SHIFT if shift else KEYMAP):
                    face.auto_expr = False
                    name = (KEYMAP_SHIFT if shift else KEYMAP)[kname]
                    face.set_expression(name)
                    print(f"Expression: {name}")

                # M: toggle manual gaze
                elif ev.key == pygame.K_m:
                    face.toggle_manual()

                # Arrow keys: nudge gaze
                elif ev.key == pygame.K_LEFT:  face.nudge(-0.12, 0)
                elif ev.key == pygame.K_RIGHT: face.nudge( 0.12, 0)
                elif ev.key == pygame.K_UP:    face.nudge(0, -0.12)
                elif ev.key == pygame.K_DOWN:  face.nudge(0,  0.12)

                # R: recenter (also clears mouse target)
                elif ev.key == pygame.K_r:
                    face.recenter()

                # Space / B: manual blink
                elif ev.key in (pygame.K_SPACE, pygame.K_b):
                    face.trigger_blink()

                # E / D: energy up/down
                elif ev.key == pygame.K_e: face.change_energy(+0.08)
                elif ev.key == pygame.K_d: face.change_energy(-0.08)

        face.update(dt)
        face.render_to(screen)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()