# face_128x64_9_live.py
# Pixar-quality Cozmo-style eyes — 128x64 px — pygame
# Architecture: EyeRenderer | BlinkController | GazeController | ExpressionController
#               + FaceTracker | MicMonitor | EmotionEngine
#
# v9: +Camera face tracking — eyes follow your real face in real time
#     +Microphone monitoring — volume/speech drives energy & arousal
#     +EmotionEngine — fuses face presence, distance, motion, mic level
#       into a live valence/arousal signal that picks expressions from:
#       angry, annoyed, awe, focused_determined, frustrated_bored, furious,
#       glee, happy, neutral, neutral_big, sad_looking_down, scared,
#       skeptical_left/right, sleepy_left/right, squint,
#       suspicious_left/right, unimpressed_left/right, worried,
#       hearing_left, hearing_right
#     +Three gaze modes: AUTO ↔ MOUSE ↔ CAMERA (C key toggles camera)
#     +Graceful degradation: runs fine with no camera / no microphone
# =============================================================================

import math, random, time, json, os, glob, threading, collections
import numpy as np
import pygame

# Optional heavy deps — imported with graceful fallback
try:
    import cv2
    _CV2_OK = True
except ImportError:
    _CV2_OK = False
    print("[WARN] opencv-python not found — camera tracking disabled")

try:
    import ctypes, ctypes.util
    _ALSA_LIB = ctypes.util.find_library("asound")
    _ALSA_OK  = bool(_ALSA_LIB)
    if _ALSA_OK:
        # Silence ALSA error spam to stderr
        _alsa = ctypes.CDLL(_ALSA_LIB)
        _ERR_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int,
                                     ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
        _alsa.snd_lib_error_set_handler(_ERR_FUNC(lambda *_: None))
except Exception:
    _ALSA_OK = False

try:
    import sounddevice as sd
    sd.query_devices()          # raises if PortAudio missing
    _SD_OK = True
except Exception:
    _SD_OK = False

# ---------------------------------------------------------------------------
# 1. CONSTANTS  (no magic numbers in rendering logic)
# ---------------------------------------------------------------------------
FB_W,  FB_H  = 128, 64
FPS_TARGET   = 60
EYE_COLOR    = (0, 167, 222)
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
# Camera & Microphone constants  (v9)
# ---------------------------------------------------------------------------
CAM_INDEX              = 0        # cv2 camera index (0 = default laptop cam)
CAM_FRAME_W            = 320      # capture resolution — small = fast detection
CAM_FRAME_H            = 240
CAM_FPS                = 15       # camera poll rate (thread, independent of render)
CAM_SCALE_DEADZONE     = 0.04     # face-centre jitter below this norm-dist ignored
CAM_ABSENT_TIMEOUT     = 2.5      # seconds without face → return to AUTO mode
CAM_TRACKING_OMEGA     = 28.0     # spring stiffness while camera-tracking
# Face distance → arousal mapping
CAM_FACE_CLOSE_NORM    = 0.45     # normalised face width at "close/excited" distance
CAM_FACE_FAR_NORM      = 0.12     # normalised face width at "far/bored" distance
# Motion sensitivity for "surprised" trigger
CAM_MOTION_THRESH      = 0.18     # normalised face-centre jump that triggers awe/scared
CAM_MOTION_DECAY       = 0.85     # per-frame motion memory decay
# Gaze output scale: camera nx/ny are ±1 over the full frame but the gaze
# spring only moves eyes ±range_x px (≈7) inside the eye socket.
# Without scaling, any face off dead-centre pegs the eyes to the edge.
# 0.65 → face at 65% of frame edge = eye at full travel; feels natural.
CAM_GAZE_SCALE         = 0.65

# Microphone constants
MIC_SAMPLE_RATE        = 16000    # Hz
MIC_CHUNK              = 512      # frames per chunk
MIC_HISTORY_SEC        = 0.8      # rolling RMS window
MIC_NOISE_FLOOR        = 0.003    # below this → silence
MIC_SPEECH_THRESH      = 0.018    # above this → talking
MIC_LOUD_THRESH        = 0.055    # above this → loud / excited
MIC_SILENCE_TIMEOUT    = 3.0      # seconds of silence → sad/bored nudge
MIC_SPEECH_COOL        = 0.4      # min seconds between speech-triggered mood bumps

# EmotionEngine blending
EMO_VALENCE_SLEW       = 0.015    # per-frame slew toward target valence
EMO_AROUSAL_SLEW       = 0.020    # per-frame slew toward target arousal
EMO_FACE_WEIGHT        = 0.65     # how strongly camera data drives emotion
EMO_MIC_WEIGHT         = 0.35     # how strongly microphone data drives emotion


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
            f"Make sure the folder 'References/cozmo_face_json/' exists "
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

        # ── Camera-target mode (v9) ────────────────────────────────────────
        # Camera mode continuously updates the spring target each frame
        # (unlike mouse mode which is click-triggered)
        self.cam_mode    = False      # True = eyes follow camera face
        self._cam_nx     = 0.0       # last known camera target (normalised)
        self._cam_ny     = 0.0
        self._cam_present= False

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

    def set_camera_target(self, nx: float, ny: float, present: bool):
        """
        Called every frame by CozmoFace when cam_mode is active.
        Continuously steers the spring toward the detected face position.
        No hold timer — stays live as long as cam_mode is True and face present.

        nx, ny arrive in −1…+1 (full camera field).
        We scale them down because the gaze spring drives a pixel offset
        of only ±range_x (≈7px) and ±range_y (≈4px).  Feeding raw ±1
        pegs the eyes to the extreme corners even for modest head movement.
        CAM_GAZE_SCALE damps the output so ±0.5 face offset → ±0.5 gaze.
        """
        scaled_nx = clamp(nx * CAM_GAZE_SCALE, -1.0 + TARGET_MARGIN_X,
                          1.0 - TARGET_MARGIN_X)
        scaled_ny = clamp(ny * CAM_GAZE_SCALE, -1.0 + TARGET_MARGIN_Y,
                          1.0 - TARGET_MARGIN_Y)
        self._cam_nx      = scaled_nx
        self._cam_ny      = scaled_ny
        self._cam_present = present

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
        elif self.cam_mode:
            # Camera mode: continuous live tracking toward face position
            self._update_camera(now)
        elif self._tgt_mode != "NONE":
            self._update_target(now)
        else:
            self._update_saccades(now, e)

        # Asymmetry heartbeat
        if now >= self._next_asym:
            self._asym_bias = random.gauss(0.0, 0.04)
            self._next_asym = now + random.uniform(2.5, 5.5)

        # Use a snappier spring when actively tracking a mouse or camera target
        if self.cam_mode and self._cam_present:
            omega = CAM_TRACKING_OMEGA
        elif self._tgt_mode in ("SNAP_IN", "HOLD"):
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

    def _update_camera(self, now: float):
        """
        Live camera-tracking gaze update called every frame.

        When face is present:
          • Spring target set directly to face position with a small
            organic alive-drift overlay so it doesn't feel robotic.
          • Dead-zone suppresses sub-pixel jitter (face bounces slightly
            even when the person is still — ignore movements < deadzone).
        When face absent:
          • Eyes search with a slow wandering saccade, then re-lock
            when the face returns.
        """
        if self._cam_present:
            # Dead-zone: compare against spring's CURRENT POSITION (.x), not its
            # target. Using .target caused the dead-zone to freeze the moment we
            # set a target, because target == _cam_nx right away. We want to
            # suppress micro-jitter around where the eye ACTUALLY IS.
            dist = math.hypot(self._cam_nx - self.gx.x,
                              self._cam_ny - self.gy.x)
            if dist > CAM_SCALE_DEADZONE:
                # Alive micro-drift overlay so tracking feels organic, not robotic
                drift_x = 0.014 * math.sin(2.0 * math.pi * 0.27 * now + 0.9)
                drift_y = 0.008 * math.sin(2.0 * math.pi * 0.17 * now + 2.1)
                self.gx.target = clamp(self._cam_nx + drift_x, -1, 1)
                self.gy.target = clamp(self._cam_ny + drift_y, -1, 1)
            self._state  = self.IDLE   # suppress saccade FSM
            self._next_t = now + 9999.0
        else:
            # No face — do a slow wandering search pattern
            if self._state == self.IDLE and now >= self._next_t:
                # Gentle slow saccade at low energy when searching
                self._start_saccade()
                self._next_t = now + random.uniform(1.2, 2.8)

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
# 9b. FACE TRACKER  — runs OpenCV Haar in a background thread
#     Outputs: face_nx, face_ny (normalised −1…+1), face_norm_w, face_present
# ---------------------------------------------------------------------------
class FaceTracker:
    """
    Background-thread face detector using cv2 Haar cascades.
    Zero external model files needed — uses bundled haarcascades.

    Outputs (thread-safe reads):
        face_nx, face_ny   — normalised face centre  (−1…+1 each axis)
        face_norm_w        — face bbox width / frame width  (0…1), proxy for distance
        face_present       — bool, face seen in last CAM_ABSENT_TIMEOUT seconds
        face_motion        — scalar, recent face-centre jump magnitude
    """

    def __init__(self):
        self.face_nx      = 0.0
        self.face_ny      = 0.0
        self.face_norm_w  = 0.0
        self.face_present = False
        self.face_motion  = 0.0
        self.running      = False
        self._lock        = threading.Lock()
        self._thread      = None
        self._last_seen   = 0.0
        self._prev_nx     = 0.0
        self._prev_ny     = 0.0

        if not _CV2_OK:
            print("[FaceTracker] OpenCV not available — tracker disabled")
            return

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            print("[FaceTracker] Haar cascade load failed — tracker disabled")
            return

        self._cap = cv2.VideoCapture(CAM_INDEX)
        if not self._cap.isOpened():
            print(f"[FaceTracker] Camera {CAM_INDEX} not accessible — tracker disabled")
            return

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_FRAME_W)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_FRAME_H)
        self._cap.set(cv2.CAP_PROP_FPS,          CAM_FPS)
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[FaceTracker] Camera {CAM_INDEX} opened  ({CAM_FRAME_W}×{CAM_FRAME_H})")

    def _loop(self):
        interval = 1.0 / CAM_FPS
        # Detect actual frame dimensions on first real frame (camera may ignore CAP_PROP_*)
        frame_w = None
        frame_h = None

        while self.running:
            t0 = time.perf_counter()
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(interval)
                continue

            # ── Use ACTUAL frame dimensions, not the requested constants ──
            # Many cameras silently ignore CAP_PROP_FRAME_WIDTH/HEIGHT requests.
            # Using hardcoded constants here was the root cause of the bottom-left
            # gaze bias: if camera returns 640×480 but we divide by 320×240, a face
            # at the true center (320, 240) maps to (-1, +1) — far bottom-left.
            fh, fw = frame.shape[:2]
            if frame_w is None:
                frame_w, frame_h = fw, fh
                print(f"[FaceTracker] Actual frame size: {frame_w}×{frame_h}")

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # equalizeHist → consistent detection across different lighting conditions
            gray  = cv2.equalizeHist(gray)
            faces = self._cascade.detectMultiScale(
                gray, scaleFactor=1.18, minNeighbors=4,
                minSize=(int(frame_w * 0.12), int(frame_h * 0.12)),
                flags=cv2.CASCADE_SCALE_IMAGE)

            now = time.time()
            if len(faces) > 0:
                # Pick largest face (closest to camera)
                x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
                cx = x + w / 2
                cy = y + h / 2
                # Normalise to −1…+1 using ACTUAL frame dimensions.
                # Mirror X: camera-left = robot-right (like a mirror, natural to viewer)
                nx = -((cx / frame_w) * 2.0 - 1.0)
                ny =  (cy / frame_h) * 2.0 - 1.0
                nw =  w  / frame_w   # normalised face width (distance proxy)

                # Motion: normalised distance from previous position
                motion = math.hypot(nx - self._prev_nx, ny - self._prev_ny)
                self._prev_nx, self._prev_ny = nx, ny

                with self._lock:
                    self.face_nx      = nx
                    self.face_ny      = ny
                    self.face_norm_w  = nw
                    self.face_present = True
                    self.face_motion  = motion
                self._last_seen = now
            else:
                # Face not found this frame
                if now - self._last_seen > CAM_ABSENT_TIMEOUT:
                    with self._lock:
                        self.face_present = False
                        self.face_motion  = 0.0

            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, interval - elapsed))

    def read(self):
        """Return a snapshot of the latest tracking state (thread-safe)."""
        with self._lock:
            return (self.face_nx, self.face_ny,
                    self.face_norm_w, self.face_present, self.face_motion)

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if hasattr(self, "_cap") and self._cap.isOpened():
            self._cap.release()


# ---------------------------------------------------------------------------
# 9c. MIC MONITOR  — rolling RMS audio level in a background thread
#     Uses sounddevice if PortAudio available, else ALSA ctypes, else silent
# ---------------------------------------------------------------------------
class MicMonitor:
    """
    Background-thread microphone RMS level monitor.
    Outputs (thread-safe reads):
        rms          — current smoothed RMS level (0…1 approx)
        is_speech    — bool, above MIC_SPEECH_THRESH
        is_loud      — bool, above MIC_LOUD_THRESH
        is_silent    — bool, below MIC_NOISE_FLOOR for MIC_SILENCE_TIMEOUT s
    """

    def __init__(self):
        self.rms        = 0.0
        self.is_speech  = False
        self.is_loud    = False
        self.is_silent  = True
        self.running    = False
        self._lock      = threading.Lock()
        self._thread    = None
        self._history   = collections.deque(maxlen=int(
                            MIC_SAMPLE_RATE / MIC_CHUNK * MIC_HISTORY_SEC))
        self._last_sound = time.time()
        self._backend   = "none"

        if _SD_OK:
            self._backend = "sounddevice"
        elif _ALSA_OK:
            self._backend = "alsa_ctypes"
        else:
            print("[MicMonitor] No audio backend available — mic monitoring disabled")
            return

        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[MicMonitor] Started with backend: {self._backend}")

    # ── sound device (PortAudio) backend ──────────────────────────────────
    def _loop_sd(self):
        import sounddevice as sd
        chunk = MIC_CHUNK
        sr    = MIC_SAMPLE_RATE
        while self.running:
            try:
                data = sd.rec(chunk, samplerate=sr, channels=1,
                              dtype="float32", blocking=True)
                rms = float(np.sqrt(np.mean(data**2)))
                self._push(rms)
            except Exception:
                time.sleep(0.1)

    # ── ALSA ctypes backend (no PortAudio needed, Linux only) ─────────────
    def _loop_alsa(self):
        """Minimal ALSA PCM capture via ctypes — 16-bit mono."""
        try:
            alsa = ctypes.CDLL(_ALSA_LIB)
            PCM_CAPTURE  = 1
            PCM_NONBLOCK = 0x00000001
            handle = ctypes.c_void_p()
            if alsa.snd_pcm_open(ctypes.byref(handle), b"default",
                                 PCM_CAPTURE, 0) < 0:
                raise RuntimeError("snd_pcm_open failed")
            if alsa.snd_pcm_set_params(handle, 2,  # SND_PCM_FORMAT_S16_LE
                                       3,           # SND_PCM_ACCESS_RW_INTERLEAVED
                                       1,           # channels
                                       MIC_SAMPLE_RATE,
                                       1,           # allow resampling
                                       ctypes.c_uint(50000)) < 0:
                raise RuntimeError("snd_pcm_set_params failed")

            buf_size = MIC_CHUNK * 2  # 16-bit = 2 bytes/sample
            buf      = ctypes.create_string_buffer(buf_size)

            while self.running:
                n = alsa.snd_pcm_readi(handle, buf, MIC_CHUNK)
                if n > 0:
                    samples = np.frombuffer(buf.raw[:n*2], dtype=np.int16)
                    rms = float(np.sqrt(np.mean((samples / 32768.0)**2)))
                    self._push(rms)
                elif n < 0:
                    alsa.snd_pcm_recover(handle, n, 0)

            alsa.snd_pcm_close(handle)
        except Exception as exc:
            print(f"[MicMonitor] ALSA backend error: {exc} — mic disabled")
            self.running = False

    def _loop(self):
        if self._backend == "sounddevice":
            self._loop_sd()
        elif self._backend == "alsa_ctypes":
            self._loop_alsa()

    def _push(self, rms: float):
        """Push a new RMS sample and update derived state."""
        self._history.append(rms)
        smooth_rms = float(np.mean(self._history)) if self._history else 0.0
        now        = time.time()
        if smooth_rms > MIC_NOISE_FLOOR:
            self._last_sound = now
        silent = (now - self._last_sound) > MIC_SILENCE_TIMEOUT

        with self._lock:
            self.rms       = smooth_rms
            self.is_speech = smooth_rms > MIC_SPEECH_THRESH
            self.is_loud   = smooth_rms > MIC_LOUD_THRESH
            self.is_silent = silent

    def read(self):
        with self._lock:
            return (self.rms, self.is_speech, self.is_loud, self.is_silent)

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=1.0)


# ---------------------------------------------------------------------------
# 9d. EMOTION ENGINE  — fuses camera + mic signals into valence / arousal
#     Drives CozmoFace.valence_t and .energy_t each frame.
#
#  Signal pipeline:
#    Camera  →  face_distance_arousal
#               face_motion_arousal (sudden moves → surprised)
#               face_absent_valence (no face → loneliness / boredom)
#    Mic     →  speech_arousal (talking → engaged)
#               loud_arousal   (loud → excited)
#               silence_valence (long silence → sad nudge)
#    Fused   →  target valence & arousal slewed toward face each frame
#    Special →  hearing_left / hearing_right when sound detected (looks toward mic)
# ---------------------------------------------------------------------------
class EmotionEngine:
    """
    Stateful emotion fusion. Call update() every frame; it modifies face.valence_t
    and face.energy_t directly via the face reference passed at construction.
    """

    def __init__(self, face, tracker: FaceTracker, mic: MicMonitor):
        self._face    = face
        self._tracker = tracker
        self._mic     = mic

        # Internal emotion state (slewed slowly)
        self._valence = 0.20
        self._arousal = 0.55

        # Cooldown timers
        self._last_speech_bump = 0.0
        self._last_motion_bump = 0.0
        self._last_expr_force  = 0.0   # when we last force-set an expression

        # Rolling face-motion smoothing (suppress single-frame glitches)
        self._motion_smooth = 0.0

        # Track whether camera mode is active
        self._cam_mode_active = False

    # ── Public entry point ─────────────────────────────────────────────────
    def update(self, dt: float, cam_mode: bool):
        """Call every frame. cam_mode=True when camera gaze mode is enabled."""
        now = time.time()
        self._cam_mode_active = cam_mode

        # ── Read sensors ──
        fnx, fny, fnw, fpresent, fmotion = self._tracker.read()
        rms, is_speech, is_loud, is_silent = self._mic.read()

        # ── Build target valence & arousal from signals ──
        tgt_valence = self._valence   # start from current state
        tgt_arousal = self._arousal

        # ─── CAMERA signals ────────────────────────────────────────────────
        if fpresent and cam_mode:
            # Face distance → arousal  (close face = high arousal = excited/alert)
            dist_arousal = clamp(
                (fnw - CAM_FACE_FAR_NORM) / (CAM_FACE_CLOSE_NORM - CAM_FACE_FAR_NORM),
                0.0, 1.0)
            # Face present → slight positive valence (someone's here!)
            tgt_valence  = lerp(tgt_valence, 0.35 + dist_arousal * 0.25,
                                EMO_FACE_WEIGHT * dt * 1.2)
            tgt_arousal  = lerp(tgt_arousal, 0.45 + dist_arousal * 0.45,
                                EMO_FACE_WEIGHT * dt * 1.5)

            # Smooth motion signal (prevent jitter-driven false positives)
            self._motion_smooth = lerp(self._motion_smooth, fmotion,
                                       1.0 - CAM_MOTION_DECAY)
            if (self._motion_smooth > CAM_MOTION_THRESH and
                    now - self._last_motion_bump > 1.5):
                # Sudden movement → surprise/awe spike
                tgt_arousal = min(1.0, tgt_arousal + 0.35)
                tgt_valence = min(1.0, tgt_valence + 0.15)
                self._last_motion_bump = now
                # Force an immediate expression for the surprise
                surprise_expr = random.choice(
                    [e for e in ("awe", "scared", "focused_determined")
                     if e in EYE_SHAPES] or [_NEUTRAL])
                if now - self._last_expr_force > 1.2:
                    self._face.set_expression(surprise_expr, through_blink=False)
                    self._last_expr_force = now

        elif not fpresent and cam_mode:
            # No face seen → lonely / bored drift
            tgt_valence = lerp(tgt_valence, -0.15, EMO_FACE_WEIGHT * dt * 0.5)
            tgt_arousal = lerp(tgt_arousal,  0.20, EMO_FACE_WEIGHT * dt * 0.5)
            self._motion_smooth = 0.0

        # ─── MICROPHONE signals ────────────────────────────────────────────
        if is_loud:
            # Loud sound → excited/high energy
            tgt_arousal = lerp(tgt_arousal, 0.85, EMO_MIC_WEIGHT * dt * 3.0)
            tgt_valence = lerp(tgt_valence, 0.40, EMO_MIC_WEIGHT * dt * 2.0)
            # Trigger "hearing" expression for dramatic loud sounds
            if (rms > MIC_LOUD_THRESH * 1.8 and
                    now - self._last_expr_force > 2.0):
                side = random.choice(["hearing_left", "hearing_right"])
                expr = side if side in EYE_SHAPES else _NEUTRAL
                self._face.set_expression(expr, through_blink=False)
                self._last_expr_force = now

        elif is_speech:
            # Talking → engaged/curious, bump energy
            if now - self._last_speech_bump > MIC_SPEECH_COOL:
                tgt_arousal = lerp(tgt_arousal, 0.62, EMO_MIC_WEIGHT * dt * 4.0)
                tgt_valence = lerp(tgt_valence, 0.25, EMO_MIC_WEIGHT * dt * 2.0)
                self._last_speech_bump = now
        elif is_silent:
            # Long silence → drift sad/bored
            tgt_arousal = lerp(tgt_arousal, 0.22, EMO_MIC_WEIGHT * dt * 0.8)
            tgt_valence = lerp(tgt_valence, -0.10, EMO_MIC_WEIGHT * dt * 0.6)

        # ── Slew internal state toward targets ──
        slew = dt * 1.0
        self._valence = lerp(self._valence, tgt_valence,
                             clamp(slew * EMO_VALENCE_SLEW * 60, 0, 0.15))
        self._arousal = lerp(self._arousal, tgt_arousal,
                             clamp(slew * EMO_AROUSAL_SLEW * 60, 0, 0.15))
        self._valence = clamp(self._valence, -1.0, 1.0)
        self._arousal = clamp(self._arousal,  0.0, 1.0)

        # ── Push to face ──
        self._face.valence_t = lerp(self._face.valence_t, self._valence, dt * 1.8)
        self._face.energy_t  = lerp(self._face.energy_t,  self._arousal, dt * 1.8)

    @property
    def valence(self): return self._valence

    @property
    def arousal(self): return self._arousal


# ---------------------------------------------------------------------------
# 10.  EXPRESSION PICKER  (mood → expression name, uses loaded JSON names)
# ---------------------------------------------------------------------------
def _pick(*candidates):
    """Return candidates that were actually loaded, or neutral."""
    available = [c for c in candidates if c in EYE_SHAPES]
    return available if available else [_NEUTRAL]

def pick_expression(valence: float, arousal: float) -> str:
    """
    Map (valence, arousal) → expression name.
    Uses all available JSON shapes including v9 additions:
    hearing_left/right, neutral_big, suspicious, unimpressed, awe, glee.
    """
    # Neutral bias — realistic idle expressiveness
    if random.random() < 0.40:
        return random.choice(_pick("neutral", "neutral_big"))

    # ── Low arousal — tired / bored ──
    if arousal < 0.18:
        return random.choice(_pick(
            "sleepy_left", "sleepy_right",
            "frustated_bored", "unimpressed_left", "unimpressed_right"))

    # ── High negative valence + high arousal — anger range ──
    if valence < -0.70 and arousal > 0.78:
        return random.choice(_pick("furious", "angry"))
    if valence < -0.55 and arousal > 0.60:
        return random.choice(_pick("angry", "furious", "annoyed"))

    # ── Medium-negative valence ──
    if valence < -0.30:
        if arousal > 0.55:
            return random.choice(_pick(
                "annoyed", "worried",
                "skeptical_left", "skeptical_right",
                "suspicious_left", "suspicious_right"))
        return random.choice(_pick(
            "worried", "sad_looking_down",
            "unimpressed_left", "unimpressed_right"))

    # ── Very high arousal + neutral valence — surprise / alert ──
    if arousal > 0.85 and abs(valence) < 0.25:
        return random.choice(_pick("scared", "awe", "focused_determined"))

    # ── High positive valence + high arousal — joy/excitement ──
    if valence > 0.70 and arousal > 0.70:
        return random.choice(_pick("glee", "awe", "happy"))

    # ── Moderate positive valence ──
    if valence > 0.35:
        return random.choice(_pick("happy", "glee"))

    # ── Focused/determined — medium arousal, neutral valence ──
    if arousal > 0.72:
        return random.choice(_pick(
            "focused_determined", "squint",
            "suspicious_left", "suspicious_right"))

    # ── Default neutral neighbourhood ──
    return random.choice(_pick("neutral", "neutral_big", "squint"))


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
    Thin orchestrator — wires all controllers together.
    v9: +FaceTracker (camera gaze), +MicMonitor, +EmotionEngine
        Three gaze modes: AUTO ↔ MOUSE (click) ↔ CAMERA (C key)
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
        self._shimmer_val  = 0
        self._shimmer_t0   = time.time() + random.uniform(2.0, IRIS_SHIMMER_PERIOD)
        self._tremor_phase_l = random.uniform(0.0, math.tau)
        self._tremor_phase_r = random.uniform(0.0, math.tau)
        self._entry_t        = -99.0
        self._entry_squeeze  = 1.0

        # ── v9: camera + mic + emotion ──────────────────────────────────
        self.tracker  = FaceTracker()
        self.mic      = MicMonitor()
        self.emotion  = EmotionEngine(self, self.tracker, self.mic)
        self.cam_mode = False    # toggled by C key

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

    def toggle_camera(self):
        """Toggle camera face-tracking gaze mode (C key)."""
        self.cam_mode = not self.cam_mode
        self.gaze.cam_mode = self.cam_mode
        if self.cam_mode:
            # Entering cam mode: disable mouse target and manual
            self.gaze.manual = False
            self.gaze.clear_target()
            print("[CAMERA MODE] ON — eyes will follow your face")
        else:
            print("[CAMERA MODE] OFF — returning to autonomous saccades")

    def nudge(self, dx, dy):
        self.gaze.nudge(dx, dy)

    def recenter(self):
        self.gaze.recenter()
        self.gaze.clear_target()

    def change_energy(self, d):
        self.energy_t = clamp(self.energy_t + d, 0, 1)

    def on_mouse_click(self, px: int, py: int):
        nx = (px / self.W) * 2.0 - 1.0
        ny = (py / self.H) * 2.0 - 1.0
        self.gaze.set_mouse_target(nx, ny)

    def on_mouse_right_click(self):
        self.gaze.clear_target()

    def cleanup(self):
        """Stop background threads gracefully on exit."""
        self.tracker.stop()
        self.mic.stop()

    # ---------------------------------------------------------------------------
    # Update
    # ---------------------------------------------------------------------------
    def update(self, dt: float):
        now = time.time()

        # ── v9: Run emotion engine first (updates valence_t, energy_t) ──
        self.emotion.update(dt, self.cam_mode)

        # ── Feed camera face position into gaze controller ──
        if self.cam_mode:
            fnx, fny, fnw, fpresent, fmotion = self.tracker.read()
            self.gaze.set_camera_target(fnx, fny, fpresent)

        # Smooth mood
        k = 1 - 0.5**(dt / 0.9)
        self.valence = lerp(self.valence, self.valence_t, k)
        self.energy  = lerp(self.energy,  self.energy_t,  k)

        # Autonomous expression — suppressed when emotion engine drives it
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
            t = age / 0.28
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

        # ---- HUD status dot (top-left corner, 3×3 px) ----
        # Green = camera tracking face   Orange = camera, no face   off = auto/mouse
        if self.cam_mode:
            _, _, _, fpresent, _ = self.tracker.read()
            dot_color = (0, 220, 80) if fpresent else (255, 140, 0)
            pygame.draw.rect(surface, dot_color, (1, 1, 3, 3))


# ---------------------------------------------------------------------------
# 12.  MAIN LOOP
# ---------------------------------------------------------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((FB_W, FB_H), pygame.NOFRAME)

    print("\n=== Cozmo Face v9 — Camera + Mic Emotion Engine ===")
    print("Keys 1-9:        ", {k: v for k,v in KEYMAP.items()})
    print("Shift+1-9:       ", {k: v for k,v in KEYMAP_SHIFT.items()})
    print("Ctrl+1-9:        ", {k: v for k,v in KEYMAP_CTRL.items()})
    print("TAB: cycle  |  A: auto-expr  |  C: camera mode  |  M: manual gaze")
    print("Arrows: nudge  |  R: recenter  |  Space/B: blink  |  E/D: energy")
    print("LEFT CLICK: set gaze target  |  RIGHT CLICK: release target")
    print("Green dot (top-left) = camera tracking face  |  Orange = camera on, no face")

    pygame.display.set_caption(
        "Cozmo v9  |  C=camera  A=auto  1-9=expr  TAB=next  "
        "E/D=energy  R=recenter  Space=blink  LClick=gaze")
    clock    = pygame.time.Clock()
    face     = CozmoFace(FB_W, FB_H)
    _tab_idx = [0]

    running = True
    try:
        while running:
            dt = clock.tick(FPS_TARGET) / 1000.0

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False

                elif ev.type == pygame.MOUSEBUTTONDOWN:
                    if ev.button == 1:
                        face.on_mouse_click(*ev.pos)
                    elif ev.button == 3:
                        face.on_mouse_right_click()

                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        running = False

                    mods  = pygame.key.get_mods()
                    shift = bool(mods & pygame.KMOD_SHIFT)
                    ctrl  = bool(mods & pygame.KMOD_CTRL)
                    kname = pygame.key.name(ev.key)

                    if ev.key == pygame.K_0:
                        face.auto_expr = True
                        face.set_expression(_NEUTRAL, instant=True)
                        face.recenter()

                    elif ev.key == pygame.K_TAB:
                        face.auto_expr = False
                        name = _ordered[_tab_idx[0] % len(_ordered)]
                        _tab_idx[0] += 1
                        face.set_expression(name)
                        print(f"Expression: {name}")

                    elif ev.key == pygame.K_a:
                        face.auto_expr = not face.auto_expr
                        print(f"Auto-expr: {'ON' if face.auto_expr else 'OFF'}")

                    # C: toggle camera face-tracking mode
                    elif ev.key == pygame.K_c:
                        face.toggle_camera()

                    elif ctrl and kname in KEYMAP_CTRL:
                        face.auto_expr = False
                        face.set_expression(KEYMAP_CTRL[kname])

                    elif kname in (KEYMAP_SHIFT if shift else KEYMAP):
                        face.auto_expr = False
                        name = (KEYMAP_SHIFT if shift else KEYMAP)[kname]
                        face.set_expression(name)
                        print(f"Expression: {name}")

                    elif ev.key == pygame.K_m:
                        face.toggle_manual()

                    elif ev.key == pygame.K_LEFT:  face.nudge(-0.12, 0)
                    elif ev.key == pygame.K_RIGHT: face.nudge( 0.12, 0)
                    elif ev.key == pygame.K_UP:    face.nudge(0, -0.12)
                    elif ev.key == pygame.K_DOWN:  face.nudge(0,  0.12)

                    elif ev.key == pygame.K_r:
                        face.recenter()

                    elif ev.key in (pygame.K_SPACE, pygame.K_b):
                        face.trigger_blink()

                    elif ev.key == pygame.K_e: face.change_energy(+0.08)
                    elif ev.key == pygame.K_d: face.change_energy(-0.08)

            face.update(dt)
            face.render_to(screen)
            pygame.display.flip()

    finally:
        face.cleanup()
        pygame.quit()


if __name__ == "__main__":
    main()