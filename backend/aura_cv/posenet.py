"""AURA hand-pose network (port of src/perception/handPoseNet.js; parity-tested).

A small MLP trained in this project (training/train_handpose.py) on MediaPipe hand
landmarks extracted from HaGRIDv2 images. Input: pose_vector(display landmarks), 63 values
(21 points relative to the wrist, divided by the image palm size). Output: softmax over
the app's static vocabulary + OTHER. The training script imports pose_vector from here, so
training and inference use exactly the same features.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

POSE_DIM = 63
_PALM_MCPS = (5, 9, 17)


def pose_vector(disp: np.ndarray | None) -> np.ndarray | None:
    """disp: (21, 3) display-space landmarks (aspect-corrected, optionally mirrored)."""
    if disp is None or getattr(disp, "shape", None) != (21, 3):
        return None
    w = disp[0]
    size = sum(float(np.linalg.norm(disp[i] - w)) for i in _PALM_MCPS) / 3.0
    if not size > 1e-6:
        return None
    v = ((disp - w) / size).reshape(-1)
    return v if np.all(np.isfinite(v)) else None


class HandPoseNet:
    def __init__(self, data: dict) -> None:
        if not isinstance(data, dict) or data.get("format") != "aura-handpose-mlp":
            raise ValueError("not an aura-handpose-mlp weights file")
        self.classes = list(data["classes"])
        self.mean = np.asarray(data["inputMean"], dtype=np.float64)
        self.std = np.asarray(data["inputStd"], dtype=np.float64)
        if self.mean.shape != (POSE_DIM,) or self.std.shape != (POSE_DIM,):
            raise ValueError("weights: input normalisation must have 63 values")
        self.layers = []
        dim = POSE_DIM
        for k, l in enumerate(data["layers"]):
            W = np.asarray(l["W"], dtype=np.float64)
            b = np.asarray(l["b"], dtype=np.float64)
            if l["in"] != dim or W.size != l["in"] * l["out"] or b.size != l["out"]:
                raise ValueError(f"weights: layer {k} has inconsistent shapes")
            self.layers.append((W.reshape(l["out"], l["in"]), b))
            dim = l["out"]
        if dim != len(self.classes):
            raise ValueError("weights: output size does not match classes")
        self.info = {"version": data.get("version"), "dataset": data.get("dataset"), "metrics": data.get("metrics"),
                     "params": int(sum(W.size + b.size for W, b in self.layers))}

    @classmethod
    def load(cls, path: str | Path) -> "HandPoseNet":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def logits(self, X: np.ndarray) -> np.ndarray:
        x = (np.atleast_2d(X) - self.mean) / self.std
        for k, (W, b) in enumerate(self.layers):
            x = x @ W.T + b
            if k < len(self.layers) - 1:
                x = np.maximum(x, 0.0)
        return x

    def predict(self, v) -> dict:
        z = self.logits(np.asarray(v, dtype=np.float64))[0]
        e = np.exp(z - z.max())
        p = e / e.sum()
        i = int(np.argmax(p))
        return {"label": self.classes[i], "score": float(p[i]), "scores": {c: float(p[j]) for j, c in enumerate(self.classes)}}
