"""Train the AURA hand-pose network on HaGRIDv2 hand landmarks (pure numpy, CPU).

Pipeline:  fetch_dataset.py -> extract_landmarks.py -> train_handpose.py
Output:    public/models/aura_handpose.json (weights + metrics) and
           training/report.json (the same metrics, kept with the code)

Model: MLP 63 -> 128 -> 64 -> C (ReLU, softmax), Adam, class-weighted cross-entropy,
L2 weight decay, early stopping on the validation loss. Inputs are posenet.pose_vector of the
display landmarks, so training and inference use the same features (both engines).
Augmentation (training split only): random horizontal flip (either hand / camera mirroring),
in-plane rotation +-25 deg, landmark jitter.

Evaluation on the held-out split:
  * the network alone (accuracy, per-class precision / recall, confusion matrix)
  * the full app decision (the Python engine's StaticGestureClassifier, i.e. the same fusion as
    the browser) in three configurations: rules only, canned model + rules (before this
    network), canned model + network + rules (now), with the canned model output recorded
    for each image by extract_landmarks.py.
Split: random, stratified, 80 / 20 with a fixed seed. The mirror has no subject ids, so the
same person may appear in both splits; real-world accuracy can be lower (see training/README.md).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from aura_cv.config import DEFAULT_CONFIG  # noqa: E402
from aura_cv.features import extract_features  # noqa: E402
from aura_cv.gestures import StaticGestureClassifier  # noqa: E402
from aura_cv.posenet import HandPoseNet, pose_vector  # noqa: E402

# HaGRID class -> app class (None = not used). THREE_FINGERS is the class whose images show
# index + middle + ring raised (checked visually and by finger extension, see README.md).
CLASS_MAP = {
    "palm": "OPEN_PALM", "stop": "OPEN_PALM", "fist": "FIST", "one": "POINT", "peace": "TWO_FINGERS",
    "two_up": "TWO_FINGERS", "like": "THUMB_UP", "dislike": "THUMB_DOWN",
    "no_gesture": "OTHER", "four": "OTHER", "ok": "OTHER", "call": "OTHER", "rock": "OTHER", "grabbing": "OTHER",
    "three_gun": "OTHER", "little_finger": "OTHER", "timeout": "OTHER", "mute": "OTHER", "point": "OTHER",
}
CLASSES = ["OPEN_PALM", "FIST", "POINT", "TWO_FINGERS", "THREE_FINGERS", "THUMB_UP", "THUMB_DOWN", "OTHER"]
APP_COMMAND_GESTURES = {"OPEN_PALM", "FIST", "POINT", "TWO_FINGERS", "THREE_FINGERS", "THUMB_UP", "THUMB_DOWN", "PINCH"}


def disp_points(lm: np.ndarray, aspect: float) -> np.ndarray:
    d = lm.astype(np.float64).copy()
    d[:, 0] *= aspect
    d[:, 2] *= aspect
    return d


def augment(X: np.ndarray, rng: np.random.Generator, copies: int) -> np.ndarray:
    out = [X]
    P = X.reshape(len(X), 21, 3)
    for _ in range(copies):
        Q = P.copy()
        flip = rng.random(len(Q)) < 0.5
        Q[flip, :, 0] *= -1
        a = np.deg2rad(rng.uniform(-25, 25, len(Q)))
        c, s = np.cos(a)[:, None], np.sin(a)[:, None]
        x, y = Q[:, :, 0].copy(), Q[:, :, 1].copy()
        Q[:, :, 0], Q[:, :, 1] = c * x - s * y, s * x + c * y
        Q += rng.normal(0, 0.03, Q.shape)
        Q -= Q[:, :1, :]  # keep the wrist at the origin, like pose_vector
        out.append(Q.reshape(len(Q), -1))
    return np.concatenate(out)


class MLP:
    def __init__(self, sizes, rng):
        self.W = [rng.normal(0, np.sqrt(2 / i), (o, i)) for i, o in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(o) for o in sizes[1:]]
        self.m = [np.zeros_like(p) for p in self.W + self.b]
        self.v = [np.zeros_like(p) for p in self.W + self.b]
        self.step = 0

    def forward(self, X):
        acts = [X]
        for k, (W, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ W.T + b
            acts.append(np.maximum(z, 0) if k < len(self.W) - 1 else z)
        return acts

    def loss_grad(self, X, y, w, l2):
        acts = self.forward(X)
        z = acts[-1]
        z = z - z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        n = len(X)
        sw = w[y]
        loss = -(sw * np.log(p[np.arange(n), y] + 1e-12)).sum() / sw.sum()
        g = p.copy()
        g[np.arange(n), y] -= 1
        g *= (sw / sw.sum())[:, None]
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for k in range(len(self.W) - 1, -1, -1):
            gW[k] = g.T @ acts[k] + l2 * self.W[k]
            gb[k] = g.sum(0)
            if k:
                g = (g @ self.W[k]) * (acts[k] > 0)
        return loss, gW + gb

    def adam(self, grads, lr, b1=0.9, b2=0.999):
        self.step += 1
        params = self.W + self.b
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g * g
            mh = self.m[i] / (1 - b1 ** self.step)
            vh = self.v[i] / (1 - b2 ** self.step)
            p -= lr * mh / (np.sqrt(vh) + 1e-8)

    def predict(self, X):
        return self.forward(X)[-1].argmax(1)


def prf(y, p, classes):
    rows = {}
    for i, c in enumerate(classes):
        tp = int(((p == i) & (y == i)).sum())
        fp = int(((p == i) & (y != i)).sum())
        fn = int(((p != i) & (y == i)).sum())
        rows[c] = {"n": int((y == i).sum()), "precision": round(tp / (tp + fp), 4) if tp + fp else None,
                   "recall": round(tp / (tp + fn), 4) if tp + fn else None}
    return rows


def app_decisions(lms, aspects, mlabels, mscores, net):
    cfg = DEFAULT_CONFIG["gestures"]
    out = {"rules": [], "model+rules": [], "model+net+rules": []}
    clf_rules = StaticGestureClassifier({**cfg, "model": {**cfg["model"], "enabled": False}, "net": {**cfg["net"], "enabled": False}})
    clf_prev = StaticGestureClassifier({**cfg, "net": {**cfg["net"], "enabled": False}})
    clf_now = StaticGestureClassifier(cfg)
    for lm, a, ml, ms in zip(lms, aspects, mlabels, mscores):
        model = {"label": ml or "None", "score": float(ms) if ml else 1.0, "scores": {}}
        f = extract_features(lm.astype(np.float64), None, aspect=float(a), mirror=True, score=None, model=model)
        if not f.valid:
            for k in out:
                out[k].append("UNKNOWN")
            continue
        out["model+net+rules"].append(_classify(clf_now, f, net))
        f.net = None
        out["model+rules"].append(clf_prev.classify(f).gesture)
        out["rules"].append(clf_rules.classify(f).gesture)
    return out


def _classify(clf, f, net):
    f.net = net.predict(f.pose) if f.pose is not None else None
    return clf.classify(f).gesture


def summarize_app(truth, decided):
    """truth: app class or OTHER. A decision is correct if it equals the class (OTHER: no command
    gesture, i.e. UNKNOWN / NONE / MOVING). PINCH on an OTHER image counts as a false activation."""
    res = {}
    for name, dec in decided.items():
        correct = sum((d == t) if t != "OTHER" else (d not in APP_COMMAND_GESTURES) for t, d in zip(truth, dec))
        pos = [i for i, t in enumerate(truth) if t != "OTHER"]
        neg = [i for i, t in enumerate(truth) if t == "OTHER"]
        per = {}
        for c in CLASSES[:-1]:
            idx = [i for i, t in enumerate(truth) if t == c]
            per[c] = round(sum(dec[i] == c for i in idx) / len(idx), 4) if idx else None
        res[name] = {
            "accuracy": round(correct / len(truth), 4),
            "recall_by_class": per,
            "unknown_on_gestures": round(sum(dec[i] not in APP_COMMAND_GESTURES for i in pos) / max(1, len(pos)), 4),
            "wrong_gesture_on_gestures": round(sum(dec[i] in APP_COMMAND_GESTURES and dec[i] != truth[i] for i in pos) / max(1, len(pos)), 4),
            "false_activation_on_other": round(sum(dec[i] in APP_COMMAND_GESTURES for i in neg) / max(1, len(neg)), 4),
        }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/hagrid/landmarks.npz"))
    ap.add_argument("--three", default="three", help="HaGRID class used as THREE_FINGERS")
    ap.add_argument("--out", default=str(ROOT / "public/models/aura_handpose.json"))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    cmap = dict(CLASS_MAP)
    for k in ("three", "three2", "three3"):
        cmap[k] = "THREE_FINGERS" if k == a.three else "OTHER"

    d = np.load(a.data)
    keep = np.array([cmap.get(str(l)) is not None for l in d["label"]])
    lms, asp, lab = d["landmarks"][keep], d["aspect"][keep], d["label"][keep]
    mlab, msc = d["model_label"][keep], d["model_score"][keep]
    X, ok = [], []
    for lm, s in zip(lms, asp):
        v = pose_vector(disp_points(lm, float(s)))
        ok.append(v is not None)
        X.append(v if v is not None else np.zeros(63))
    ok = np.array(ok)
    X = np.asarray(X)[ok]
    lms, asp, lab, mlab, msc = lms[ok], asp[ok], lab[ok], mlab[ok], msc[ok]
    y = np.array([CLASSES.index(cmap[str(l)]) for l in lab])
    rng = np.random.default_rng(a.seed)

    # stratified split
    val = np.zeros(len(y), bool)
    for c in range(len(CLASSES)):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        val[idx[: max(1, len(idx) // 5)]] = True
    Xtr, ytr, Xva, yva = X[~val], y[~val], X[val], y[val]
    Xtr_aug = augment(Xtr, rng, copies=4)
    ytr_aug = np.tile(ytr, 5)
    mean, std = Xtr_aug.mean(0), Xtr_aug.std(0) + 1e-6
    norm = lambda Z: (Z - mean) / std  # noqa: E731
    counts = np.bincount(ytr, minlength=len(CLASSES)).astype(float)
    w = counts.sum() / (len(CLASSES) * np.maximum(counts, 1))

    net = MLP([63, 128, 64, len(CLASSES)], rng)
    best = (np.inf, None, 0)
    t0 = time.time()
    A, Yv = norm(Xtr_aug), norm(Xva)
    for ep in range(a.epochs):
        lr = 2e-3 * (0.5 * (1 + np.cos(np.pi * ep / a.epochs)))
        perm = rng.permutation(len(A))
        for i in range(0, len(A), 256):
            b = perm[i:i + 256]
            _, g = net.loss_grad(A[b], ytr_aug[b], w, 1e-4)
            net.adam(g, lr)
        vl, _ = net.loss_grad(Yv, yva, w, 0.0)
        if vl < best[0]:
            best = (vl, ([W.copy() for W in net.W], [b.copy() for b in net.b]), ep)
        if ep % 10 == 0 or ep == a.epochs - 1:
            acc = (net.predict(Yv) == yva).mean()
            print(f"epoch {ep:3d}  val loss {vl:.4f}  val acc {acc:.4f}", flush=True)
    net.W, net.b = best[1]
    pv = net.predict(Yv)
    conf = np.zeros((len(CLASSES), len(CLASSES)), int)
    for t, p in zip(yva, pv):
        conf[t, p] += 1

    weights = {
        "format": "aura-handpose-mlp", "version": 1, "classes": CLASSES,
        "inputMean": [round(float(v), 7) for v in mean], "inputStd": [round(float(v), 7) for v in std],
        "layers": [{"in": int(W.shape[1]), "out": int(W.shape[0]), "W": [round(float(v), 7) for v in W.reshape(-1)],
                    "b": [round(float(v), 7) for v in b]} for W, b in zip(net.W, net.b)],
        "dataset": {"name": "HaGRIDv2 (512 px mirror testdummyvt/hagRIDv2_512px, test split, one shard per class)",
                    "licence": "HaGRID licence (variant of CC BY-SA 4.0), https://github.com/hukenovs/hagrid/blob/master/license/en_us.pdf",
                    "classMap": cmap, "samples": int(len(y)), "train": int(len(ytr)), "val": int(len(yva)),
                    "perClass": {CLASSES[c]: int(n) for c, n in enumerate(np.bincount(y, minlength=len(CLASSES)))}},
    }
    rt = HandPoseNet(weights)  # the exported file, through the runtime code path
    rt_pred = np.array([CLASSES.index(rt.predict(v)["label"]) for v in Xva])
    assert (rt_pred == pv).mean() > 0.999, "exported weights disagree with the trained network"

    truth = [CLASSES[c] for c in yva]
    decided = app_decisions(lms[val], asp[val], mlab[val], msc[val], rt)
    metrics = {
        "net": {"valAccuracy": round(float((pv == yva).mean()), 4), "bestEpoch": int(best[2]), "trainSeconds": round(time.time() - t0, 1),
                "perClass": prf(yva, pv, CLASSES), "confusion": {"classes": CLASSES, "rows_true_cols_pred": conf.tolist()}},
        "app": summarize_app(truth, decided),
        "split": "random stratified 80/20 (seed %d); no subject ids in the mirror" % a.seed,
    }
    weights["metrics"] = metrics
    Path(a.out).write_text(json.dumps(weights, separators=(",", ":")), encoding="utf-8")
    (ROOT / "training/report.json").write_text(json.dumps({"dataset": weights["dataset"], "metrics": metrics}, indent=1), encoding="utf-8")
    print(json.dumps(metrics, indent=1))
    print("saved", a.out, f"{Path(a.out).stat().st_size / 1024:.0f} KiB", "params", rt.info["params"])


if __name__ == "__main__":
    sys.exit(main())
