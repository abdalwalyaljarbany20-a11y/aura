"""Deterministic gesture recognition (Python port of src/gestures/*.js).

Static rules (e_x = extension of finger x, pc = ramp(pinch_distance, 0.38 -> 0.22)):

  OPEN_PALM = ramp(min(e_index, e_middle, e_ring, e_pinky), 0.6 -> 0.9) * (1 - pc)
  FIST      = min(1-e_index, 1-e_middle, 1-e_ring, 1-e_pinky) * (1 - ramp(e_thumb, 0.7 -> 0.95))
  POINT     = min(e_index, 1-e_middle, 1-e_ring, 1-e_pinky) * (1 - pc)
  PINCH     = pc * ramp(e_index, 0.1 -> 0.35)
  TWO_FINGERS = ramp(min(e_index, e_middle), 0.6 -> 0.9) * ramp(max(e_ring, e_pinky), 0.9 -> 0.7)
                * (1 - pc) * (1 - ramp(e_thumb, 0.75 -> 1.0))   (walk; joint geometry, not fingertip spread;
                folded = not clearly straight, calibrated on a real frontal V sign: ring 0.43, pinky 0.66)
  THUMB_UP / THUMB_DOWN = fold4 * ramp(e_thumb, 0.7 -> 0.95) * ramp(-/+ thumb_dir.y, 0.5 -> 0.8) * (1 - pc),
                fold4 = ramp(max(e_index, e_middle, e_ring, e_pinky), 0.95 -> 0.75)

Scores are rule memberships in [0, 1], not calibrated probabilities. The winner must
reach ``minScore`` and beat the runner-up by ``minMargin``; otherwise UNKNOWN.

Temporal logic:
  * motion gate: requireStill gestures become MOVING above ``maxStillSpeed`` palm/s
  * GestureStabilizer: enter after continuous ``holdMs``, exit after ``exitMs`` absence;
    exactly one ``start`` per continuous hold (reset condition: release + new hold)
  * SwipeDetector: timestamped palm-centre history (see class docstring); suppressed while
    pinching, pointing or walking (PINCH / POINT / TWO_FINGERS raw or stable)
  * WalkCadenceDetector / SpinDetector: continuous motion gestures FINGER_WALK and SPIN
    (start/end events + result["motion"]); walking is evaluated first and blocks a spin
    start (and vice versa), so a tie in one frame goes to walking
  * tracking loss releases the stable gesture immediately and clears swipe history
"""

from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import dataclass, field

from .config import STATIC_GESTURES
from .features import HandFeatures, ramp

PINCH_RANGE = (0.38, 0.22)
# Deep-learning gesture classes (MediaPipe GestureRecognizer) -> app gestures (port of MODEL_LABELS)
MODEL_LABELS = {"Open_Palm": "OPEN_PALM", "Closed_Fist": "FIST", "Pointing_Up": "POINT",
                "Thumb_Up": "THUMB_UP", "Thumb_Down": "THUMB_DOWN", "Victory": "TWO_FINGERS"}
RULE_ASSIST = ("POINT", "TWO_FINGERS", "THREE_FINGERS")
# shapes the project-trained hand-pose net may report when the canned model is unsure (port of NET_ASSIST)
NET_ASSIST = frozenset({"POINT", "TWO_FINGERS", "THREE_FINGERS", "OPEN_PALM"})
# canned labels a three-finger hand is read as (port of THREE_OVERRIDABLE)
THREE_OVERRIDABLE = frozenset({None, "TWO_FINGERS", "OPEN_PALM"})
# geometric agreement required before these model classes are accepted (port of MODEL_AGREEMENT)
MODEL_AGREEMENT = {"THUMB_UP": 0.2, "THUMB_DOWN": 0.2}
NEUTRAL = frozenset({"NONE", "UNKNOWN", "MOVING"})


@dataclass
class Classification:
    gesture: str
    confidence: float
    scores: dict = field(default_factory=dict)
    reason: str | None = None
    source: str = "rules"


class StaticGestureClassifier:
    def __init__(self, cfg: dict) -> None:
        m = cfg.get("model", {})
        self.model_enabled = m.get("enabled", True)
        self.model_min_score = m.get("minScore", 0.6)
        n = cfg.get("net", {})
        self.net_enabled = n.get("enabled", True)
        self.net_min_score = n.get("minScore", 0.7)
        self.min_score = cfg["minScore"]
        self.min_margin = cfg["minMargin"]
        self.min_tracking_score = cfg["minTrackingScore"]
        self.min_in_frame_ratio = cfg["minInFrameRatio"]

    @staticmethod
    def scores(f: HandFeatures) -> dict[str, float]:
        e = f.extension
        pc = ramp(f.pinch_distance, *PINCH_RANGE)
        four = (e("index"), e("middle"), e("ring"), e("pinky"))
        fold4 = ramp(max(four), 0.95, 0.75)  # real side-on thumbs-up: curled ring measured 0.82
        thumb = fold4 * ramp(e("thumb"), 0.7, 0.95) * (1 - pc)
        return {
            "OPEN_PALM": ramp(min(four), 0.6, 0.9) * (1 - pc),
            "FIST": min(1 - x for x in four) * (1 - ramp(e("thumb"), 0.7, 0.95)),
            "POINT": min(e("index"), 1 - e("middle"), 1 - e("ring"), 1 - e("pinky")) * (1 - pc),
            "PINCH": pc * ramp(e("index"), 0.1, 0.35),
            "TWO_FINGERS": ramp(min(e("index"), e("middle")), 0.6, 0.9) * ramp(max(e("ring"), e("pinky")), 0.9, 0.7)
            * (1 - pc) * (1 - ramp(e("thumb"), 0.75, 1.0)),
            "THREE_FINGERS": ramp(min(e("index"), e("middle"), e("ring")), 0.6, 0.9) * ramp(e("pinky"), 0.9, 0.7)
            * (1 - pc) * (1 - ramp(e("thumb"), 0.75, 1.0)),
            "THUMB_UP": thumb * ramp(-f.thumb_dir[1], 0.5, 0.8),
            "THUMB_DOWN": thumb * ramp(f.thumb_dir[1], 0.5, 0.8),
        }

    def classify(self, f: HandFeatures | None, enabled=None) -> Classification:
        if f is None or not f.valid:
            return Classification("NONE", 0.0, {}, f.reason if f else "no hand")
        if f.score is not None and f.score < self.min_tracking_score:
            return Classification("UNKNOWN", 0.0, {}, "low tracking confidence")
        if f.in_frame_ratio < self.min_in_frame_ratio:
            return Classification("UNKNOWN", 0.0, {}, "hand partly outside the image")
        all_scores = self.scores(f)
        allowed = set(enabled) if enabled is not None else None
        net = f.net if self.net_enabled and isinstance(getattr(f, "net", None), dict) else None
        if self.model_enabled and f.model is not None:
            return self._fuse(f.model, all_scores, allowed, net)
        ranked = sorted(((g, s) for g, s in all_scores.items() if allowed is None or g in allowed),
                        key=lambda gs: gs[1], reverse=True)
        if not ranked:
            return Classification("UNKNOWN", 0.0, all_scores, "no gestures enabled")
        best, best_score = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        if best_score < self.min_score:
            return Classification("UNKNOWN", best_score, all_scores, "no rule matched strongly")
        if best_score - second < self.min_margin:
            return Classification("UNKNOWN", best_score, all_scores, "ambiguous between gestures")
        return Classification(best, best_score, all_scores, None)

    def _fuse(self, model: dict, rules: dict, allowed, net: dict | None = None) -> Classification:
        """Model / rule fusion (port of StaticGestureClassifier._fuse; same order and outputs)."""
        def ok(g):
            return allowed is None or g in allowed

        scores = dict(rules)
        for cls, g in MODEL_LABELS.items():
            if cls in (model.get("scores") or {}):
                scores[g] = model["scores"][cls]
        if ok("PINCH") and rules["PINCH"] >= self.min_score:
            return Classification("PINCH", rules["PINCH"], scores, None, "pinch-distance")
        g = MODEL_LABELS.get(model["label"])
        weak = not g or model["score"] < self.model_min_score
        if ok("THREE_FINGERS") and (weak or g in THREE_OVERRIDABLE):
            if net and net["label"] == "THREE_FINGERS" and net["score"] >= self.net_min_score:
                return Classification("THREE_FINGERS", net["score"], scores, None, f"aura-net (model: {model['label']})")
            if not net and rules["THREE_FINGERS"] >= max(self.min_score, 0.8):
                return Classification("THREE_FINGERS", rules["THREE_FINGERS"], scores, None, f"rules (model: {model['label']})")
        agrees = not g or g not in MODEL_AGREEMENT or (rules["THUMB_UP"] + rules["THUMB_DOWN"]) >= MODEL_AGREEMENT[g]
        if g and ok(g) and model["score"] >= self.model_min_score and agrees:
            return Classification(g, model["score"], scores, None, "model")
        if g and not agrees:
            return Classification("UNKNOWN", model["score"], scores,
                                  f"model: {model['label']} {model['score']:.2f} rejected (hand geometry disagrees)", "model")
        if weak:
            if net and net["label"] in NET_ASSIST and ok(net["label"]) and net["score"] >= self.net_min_score:
                return Classification(net["label"], net["score"], scores, None, f"aura-net (model: {model['label']})")
            net_sure = bool(net) and net["score"] >= self.net_min_score
            ranked = [] if net_sure else sorted(((k, v) for k, v in rules.items() if k in RULE_ASSIST and ok(k)), key=lambda kv: kv[1], reverse=True)
            if ranked and ranked[0][1] >= max(self.min_score, 0.8):
                return Classification(ranked[0][0], ranked[0][1], scores, None, f"rules (model: {model['label']})")
        return Classification("UNKNOWN", model["score"], scores, f"model: {model['label']} {model['score']:.2f}", "model")


class GestureStabilizer:
    """Hysteresis: enter after a continuous hold, exit after a sustained absence."""

    def __init__(self, exit_ms: float, hold_ms) -> None:
        self.exit_ms = exit_ms
        self.hold_ms = hold_ms  # callable gesture -> ms
        self.reset()

    def reset(self) -> None:
        self.stable = "NONE"
        self.stable_since = 0.0
        self.last_seen_stable = 0.0
        self.stable_confidence = 0.0
        self.candidate: str | None = None
        self.candidate_since = 0.0

    def update(self, label: str, confidence: float, t: float) -> list[dict]:
        events: list[dict] = []
        if label == self.stable and label not in NEUTRAL:
            self.last_seen_stable = t
            self.stable_confidence = confidence
            self.candidate = None
            return events
        if self.stable not in NEUTRAL and t - self.last_seen_stable >= self.exit_ms:
            events.append({"type": "end", "gesture": self.stable, "t": t,
                           "durationMs": self.last_seen_stable - self.stable_since})
            self.stable = "NONE"
            self.stable_confidence = 0.0
        if label in NEUTRAL:
            self.candidate = None
            return events
        if self.candidate != label:
            self.candidate = label
            self.candidate_since = t
        if self.stable in NEUTRAL and t - self.candidate_since >= self.hold_ms(label):
            self.stable = label
            self.stable_since = t
            self.last_seen_stable = t
            self.stable_confidence = confidence
            self.candidate = None
            events.append({"type": "start", "gesture": label, "t": t, "confidence": confidence})
        return events

    def release(self, t: float) -> list[dict]:
        events = []
        if self.stable not in NEUTRAL:
            events.append({"type": "end", "gesture": self.stable, "t": t,
                           "durationMs": self.last_seen_stable - self.stable_since})
        self.reset()
        return events


class SwipeDetector:
    """Horizontal swipe from a timestamped palm-centre history (display space).

    Fires when, within ``windowMs``, the palm travelled >= ``minDistance`` palm lengths
    horizontally at >= ``minSpeed`` palm lengths/s with |dy| <= maxVerticalRatio*|dx|.
    History is cleared after a swipe; ``cooldownMs`` blocks all swipes, and the opposite
    direction is blocked for ``oppositeSuppressMs``; a gap > ``maxGapMs`` resets history.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = dict(cfg)
        self.history: deque[tuple[float, float, float, float]] = deque()
        self.blocked_until = float("-inf")
        self.last: tuple[str, float] | None = None

    def reset(self) -> None:
        self.history.clear()

    def update(self, t: float, center, palm: float, suppressed: bool = False) -> dict | None:
        c = self.cfg
        if not c["enabled"] or center is None or not palm > 0:
            self.reset()
            return None
        if self.history and t - self.history[-1][0] > c["maxGapMs"]:
            self.history.clear()
        self.history.append((t, center[0], center[1], palm))
        while self.history and self.history[0][0] < t - c["windowMs"]:
            self.history.popleft()
        if suppressed or t < self.blocked_until or len(self.history) < 3:
            return None

        h = list(self.history)
        now = h[-1]
        mean_palm = sum(s[3] for s in h) / len(h)
        best = None
        for s in h[:-2]:
            dt = now[0] - s[0]
            if dt <= 0:
                continue
            dx = (now[1] - s[1]) / mean_palm
            dy = (now[2] - s[2]) / mean_palm
            dist = abs(dx)
            speed = dist / (dt / 1000.0)
            if dist < c["minDistance"] or speed < c["minSpeed"] or abs(dy) > c["maxVerticalRatio"] * dist:
                continue
            if best is None or dist > best["distance"]:
                best = {"dx": dx, "distance": dist, "speed": speed, "durationMs": dt}
        if best is None:
            return None
        gesture = "SWIPE_LEFT" if best["dx"] < 0 else "SWIPE_RIGHT"
        if self.last and self.last[0] != gesture and t - self.last[1] < c["oppositeSuppressMs"]:
            self.history.clear()
            return None
        self.last = (gesture, t)
        self.blocked_until = t + c["cooldownMs"]
        self.history.clear()
        return {"gesture": gesture, "distance": best["distance"], "speed": best["speed"], "durationMs": best["durationMs"]}

    def speed(self) -> float:
        """Palm speed (palm lengths/s) over the last ~150 ms."""
        h = self.history
        if len(h) < 2:
            return 0.0
        now = h[-1]
        i = len(h) - 2
        while i > 0 and now[0] - h[i][0] < 150:
            i -= 1
        s = h[i]
        dt = (now[0] - s[0]) / 1000.0
        if dt <= 0:
            return 0.0
        return ((now[1] - s[1]) ** 2 + (now[2] - s[2]) ** 2) ** 0.5 / now[3] / dt


_MIN_SPIN_POINTS = 6
_DEG = math.pi / 180


def legs_pose_score(f: HandFeatures) -> float:
    """Legs pose for finger walking: ring, pinky, thumb folded; index or middle raised (port of legsPoseScore)."""
    if f is None or not f.valid:
        return 0.0
    e = f.extension
    return (ramp(max(e("ring"), e("pinky")), 0.9, 0.7) * (1 - ramp(e("thumb"), 0.75, 1.0))
            * ramp(max(e("index"), e("middle")), 0.3, 0.6))


class SpinDetector:
    """Continuous index-finger rotation (port of src/gestures/SpinDetector.js).

    Signature over the last windowMs (>= minWindowMs) around its centroid: mean radius >=
    minRadius, radius std/mean <= maxRadiusCv, steps <= maxStepDeg, consistency >=
    minConsistency, |sweep| >= minSweepDeg. Rate = sweep of the newest rateWindowMs / its
    duration (rad/s, + = clockwise on screen). Active after enterMs of signature, ends after
    releaseMs without it, poseGraceMs without the pointing shape, or at once on hand loss.
    Arithmetic order matches the JS implementation (identical events and rates).
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = dict(cfg)
        self.reset()

    def reset(self) -> None:
        self.history: list[tuple[float, float, float, float]] = []
        self.active = False
        self.omega = 0.0
        self.signature_since: float | None = None
        self.last_signature = float("-inf")
        self.inactive_since: float | None = None

    def update(self, t: float, tip, palm: float, pose: bool, blocked: bool = False) -> dict:
        c = self.cfg
        if not c["enabled"] or tip is None or not palm > 0:
            return self._end()
        if not pose:
            if self.inactive_since is None:
                self.inactive_since = t
            if t - self.inactive_since > c["poseGraceMs"]:
                return self._end()
            return {"active": self.active, "omega": self.omega if self.active else 0.0, "event": None}
        self.inactive_since = None
        h = self.history
        if h and t - h[-1][0] > c["maxGapMs"]:
            h.clear()
        h.append((t, tip[0], tip[1], palm))
        while h and h[0][0] < t - c["windowMs"]:
            h.pop(0)
        m = self._measure(h)
        event = None
        if m is not None:
            self.last_signature = t
            if self.signature_since is None:
                self.signature_since = t
            if self.active:
                self.omega = m["rate"]
            elif not blocked and t - self.signature_since >= c["enterMs"]:
                self.active = True
                self.omega = m["rate"]
                event = "start"
        else:
            self.signature_since = None
            if self.active and t - self.last_signature > c["releaseMs"]:
                self.active = False
                self.omega = 0.0
                event = "end"
        return {"active": self.active, "omega": self.omega if self.active else 0.0, "event": event}

    def _end(self) -> dict:
        was = self.active
        self.reset()
        return {"active": False, "omega": 0.0, "event": "end" if was else None}

    def _measure(self, h) -> dict | None:
        c = self.cfg
        n = len(h)
        if n < _MIN_SPIN_POINTS:
            return None
        last = h[-1][0]
        if last - h[0][0] < c["minWindowMs"]:
            return None
        cx = cy = palm = 0.0
        for k in range(n):
            cx += h[k][1]
            cy += h[k][2]
            palm += h[k][3]
        cx /= n
        cy /= n
        palm /= n
        r_sum = r2_sum = 0.0
        for k in range(n):
            dx = h[k][1] - cx
            dy = h[k][2] - cy
            r = math.sqrt(dx * dx + dy * dy) / palm
            r_sum += r
            r2_sum += r * r
        radius = r_sum / n
        if radius < c["minRadius"]:
            return None
        variance = max(0.0, r2_sum / n - radius * radius)
        if math.sqrt(variance) / radius > c["maxRadiusCv"]:
            return None
        total = 0.0
        pos = neg = 0
        recent = 0.0
        recent_from = None
        prev = math.atan2(h[0][2] - cy, h[0][1] - cx)
        for k in range(1, n):
            a = math.atan2(h[k][2] - cy, h[k][1] - cx)
            d = a - prev
            if d > math.pi:
                d -= 2 * math.pi
            elif d <= -math.pi:
                d += 2 * math.pi
            if abs(d) > c["maxStepDeg"] * _DEG:
                return None
            total += d
            if d > 0:
                pos += 1
            elif d < 0:
                neg += 1
            if h[k - 1][0] >= last - c["rateWindowMs"]:
                if recent_from is None:
                    recent_from = h[k - 1][0]
                recent += d
            prev = a
        if abs(total) < c["minSweepDeg"] * _DEG:
            return None
        steps = pos + neg
        consistency = ((pos if total > 0 else neg) / steps) if steps else 0.0
        if consistency < c["minConsistency"]:
            return None
        if recent_from is not None and last > recent_from:
            rate = recent / ((last - recent_from) / 1000)
        else:
            rate = total / ((last - h[0][0]) / 1000)
        return {"total": total, "radius": radius, "consistency": consistency, "rate": rate}


class WalkCadenceDetector:
    """Finger walking from alternating index / middle "legs" (port of src/gestures/WalkCadenceDetector.js).

    Over windowMs: each finger moves >= minAmplitude, the two signals are anti-correlated
    (<= maxCorrelation), steps are crossings of the difference through +-hysteresis around
    its window mean; cadence = steps/s over the last few steps, lowered to 1/(time since the
    last step) when slowing. Starts after minSteps, ends after stopMs without a step,
    poseGraceMs without the legs pose, when the signal stops qualifying, or on hand loss.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = dict(cfg)
        self.reset()

    def reset(self) -> None:
        self.history: list[tuple[float, float, float]] = []
        self.steps: list[float] = []
        self.side = 0
        self.active = False
        self.cadence = 0.0
        self.inactive_since: float | None = None

    def update(self, t: float, legs, pose: bool, blocked: bool = False) -> dict:
        c = self.cfg
        if not c["enabled"] or legs is None:
            return self._end()
        if not pose:
            if self.inactive_since is None:
                self.inactive_since = t
            if t - self.inactive_since > c["poseGraceMs"]:
                return self._end()
            return {"active": self.active, "cadence": self.cadence if self.active else 0.0, "event": None}
        self.inactive_since = None
        h = self.history
        h.append((t, legs[0], legs[1]))
        while h and h[0][0] < t - c["windowMs"]:
            h.pop(0)
        if len(self.steps) > 6:
            del self.steps[:len(self.steps) - 6]  # the last steps, for the cadence
        n = len(h)
        mi = mm = 0.0
        lo = [math.inf, math.inf]
        hi = [-math.inf, -math.inf]
        for k in range(n):
            mi += h[k][1]
            mm += h[k][2]
            lo = [min(lo[0], h[k][1]), min(lo[1], h[k][2])]
            hi = [max(hi[0], h[k][1]), max(hi[1], h[k][2])]
        mi /= n
        mm /= n
        sii = smm = sim = 0.0
        for k in range(n):
            a = h[k][1] - mi
            b = h[k][2] - mm
            sii += a * a
            smm += b * b
            sim += a * b
        corr = sim / math.sqrt(sii * smm) if sii > 0 and smm > 0 else 0.0
        moving = (hi[0] - lo[0] >= c["minAmplitude"] and hi[1] - lo[1] >= c["minAmplitude"]
                  and hi[0] >= c["minReach"] and hi[1] >= c["minReach"])  # a curled finger is not a leg
        alternating = moving and corr <= c["maxCorrelation"]
        dev = legs[0] - legs[1] - (mi - mm)
        side = 1 if dev > c["hysteresis"] else -1 if dev < -c["hysteresis"] else 0
        d_lo, d_hi = math.inf, -math.inf
        k = n - 1
        while k >= 0 and h[k][0] >= t - 200:
            d = h[k][1] - h[k][2]
            d_lo = min(d_lo, d)
            d_hi = max(d_hi, d)
            k -= 1
        stepping = d_hi - d_lo >= c["hysteresis"]  # real finger motion, not a drifting mean
        if side != 0 and side != self.side:
            if self.side != 0 and alternating and stepping:
                self.steps.append(t)
            self.side = side
        s = self.steps
        cadence = 0.0
        if len(s) >= 2:
            k0 = max(0, len(s) - 5)
            avg = ((len(s) - 1 - k0) * 1000) / (s[-1] - s[k0])
            since_last = t - s[-1]
            cadence = min(avg, 1000 / since_last) if since_last > 1000 / avg else avg
        recent = len(s) > 0 and t - s[-1] <= c["stopMs"]
        # minSteps (within the window) is a start filter; continuing needs a recent step only
        in_window = len([x for x in s if x >= t - c["windowMs"]])
        valid = alternating and recent and (self.active or in_window >= c["minSteps"]) and cadence <= c["maxStepRate"]
        event = None
        if valid:
            self.cadence = cadence
            if not self.active and not blocked:
                self.active = True
                event = "start"
        elif self.active:
            self.active = False
            self.cadence = 0.0
            event = "end"
        return {"active": self.active, "cadence": self.cadence if self.active else 0.0, "event": event}

    def _end(self) -> dict:
        was = self.active
        self.reset()
        return {"active": False, "cadence": 0.0, "event": "end" if was else None}


class GestureRecognitionEngine:
    """Static classification + motion gate + stabiliser + swipes + circles, for the primary hand."""

    def __init__(self, cfg: dict, net=None) -> None:
        self.cfg = copy.deepcopy(cfg)
        self.net = net  # optional HandPoseNet (posenet.py)
        self.classifier = StaticGestureClassifier(self.cfg)
        self.swipe = SwipeDetector(self.cfg["swipe"])
        self.spin = SpinDetector(self.cfg["spin"])
        self.walk = WalkCadenceDetector(self.cfg["walk"])
        defs = self.cfg["definitions"]
        self.stabilizer = GestureStabilizer(self.cfg["exitMs"], lambda g: defs.get(g, {}).get("holdMs", 300))
        self.enabled = [g for g in STATIC_GESTURES if defs[g]["enabled"]]
        self.suppress_static_until = float("-inf")
        self.had_hand = False

    def update(self, f: HandFeatures | None, t: float) -> dict:
        if f is None or not f.valid:
            events = self.stabilizer.release(t) if self.had_hand else []
            w = self.walk.update(t, None, False)
            sp = self.spin.update(t, None, 0.0, False)
            if w["event"]:
                events.append({"type": w["event"], "gesture": "FINGER_WALK", "t": t, "confidence": None})
            if sp["event"]:
                events.append({"type": sp["event"], "gesture": "SPIN", "t": t, "confidence": None})
            self.swipe.reset()
            self.had_hand = False
            return {
                "motion": {"spin": {"active": False, "omega": 0.0}, "walk": {"active": False, "cadence": 0.0}},
                "hand": False,
                "raw": {"gesture": "NONE", "label": "NONE", "confidence": 0.0, "scores": {},
                        "reason": f.reason if f else "no hand"},
                "stable": {"gesture": "NONE", "confidence": 0.0},
                "speed": 0.0,
                "events": events,
            }
        self.had_hand = True
        if self.net is not None and f.pose is not None:
            f.net = self.net.predict(f.pose)
        raw = self.classifier.classify(f, self.enabled)
        continuous = ("PINCH", "POINT", "TWO_FINGERS", "THREE_FINGERS")
        no_swipe = (raw.gesture in continuous or self.stabilizer.stable in continuous
                    or self.walk.active or self.spin.active)
        swipe = self.swipe.update(t, f.palm_center, f.palm_size_image, suppressed=no_swipe)
        speed = self.swipe.speed()
        w = self.walk.update(t, f.legs, legs_pose_score(f) >= 0.5, self.spin.active)
        sp = self.spin.update(t, f.index_tip, f.palm_size_image, raw.gesture == "POINT", w["active"])

        label = raw.gesture
        if self.cfg["definitions"].get(label, {}).get("requireStill") and speed > self.cfg["maxStillSpeed"]:
            label = "MOVING"
        if t < self.suppress_static_until:
            label = "NONE"
        events: list[dict] = []
        if swipe:
            events.append({"type": "event", "gesture": swipe["gesture"], "t": t, "confidence": None, "detail": swipe})
            self.suppress_static_until = t + self.cfg["postSwipeSuppressMs"]
            label = "NONE"
        if w["event"]:
            events.append({"type": w["event"], "gesture": "FINGER_WALK", "t": t, "confidence": None})
        if sp["event"]:
            events.append({"type": sp["event"], "gesture": "SPIN", "t": t, "confidence": None})
        stab = self.stabilizer.update(label, raw.confidence, t)
        events.extend(stab)
        # pinch tap: a stable pinch released (not lost) within pinchTap.maxMs (port of the JS engine)
        tap = self.cfg.get("pinchTap", {})
        for e in stab:
            if e["type"] == "end" and e["gesture"] == "PINCH" and tap.get("enabled", True) and e["durationMs"] <= tap.get("maxMs", 600):
                events.append({"type": "event", "gesture": "PINCH_TAP", "t": t, "confidence": None, "detail": {"durationMs": e["durationMs"]}})
        return {
            "motion": {"spin": {"active": sp["active"], "omega": sp["omega"]},
                       "walk": {"active": w["active"], "cadence": w["cadence"]}},
            "hand": True,
            "raw": {"gesture": raw.gesture, "label": label, "confidence": raw.confidence,
                    "scores": raw.scores, "reason": raw.reason, "source": raw.source},
            "stable": {"gesture": self.stabilizer.stable, "confidence": self.stabilizer.stable_confidence},
            "speed": speed,
            "events": events,
        }

    def reset(self) -> None:
        self.stabilizer.reset()
        self.swipe.reset()
        self.spin.reset()
        self.walk.reset()
        self.had_hand = False
        self.suppress_static_until = float("-inf")
