"""Extract MediaPipe hand landmarks from the downloaded HaGRIDv2 shards.

For every image: decode locally, run the app's own GestureRecognizer (IMAGE mode, CPU) and
keep images with exactly one detected hand (HaGRID images may also show a second, relaxed
hand; the image-level label would not say which hand gestures, so those are skipped).
Saved per sample: the 21 image landmarks (x, y, z), the image aspect, the HaGRID label, the
canned model's top class/score (to evaluate the full fusion later) and the source row.

Output: data/hagrid/landmarks.npz. Nothing leaves the machine; images are not stored.
Usage:  python training/extract_landmarks.py [--raw data/hagrid/raw] [--limit 0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
HAGRID_CLASSES = [
    "call", "dislike", "fist", "four", "grabbing", "grip", "hand_heart", "hand_heart2", "holy", "like",
    "little_finger", "middle_finger", "mute", "no_gesture", "ok", "one", "palm", "peace", "peace_inverted",
    "point", "rock", "stop", "stop_inverted", "take_picture", "three", "three2", "three3", "three_gun",
    "thumb_index", "thumb_index2", "timeout", "two_up", "two_up_inverted", "xsign",
]


def make_recognizer():
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision

    opts = vision.GestureRecognizerOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(ROOT / "public/models/gesture_recognizer.task"),
                                     delegate=mpp.BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.IMAGE, num_hands=2,
        min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5)
    return mp, vision.GestureRecognizer.create_from_options(opts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data/hagrid/raw"))
    ap.add_argument("--out", default=str(ROOT / "data/hagrid/landmarks.npz"))
    ap.add_argument("--limit", type=int, default=0, help="max images per shard (0 = all)")
    a = ap.parse_args()
    mp, rec = make_recognizer()
    L, ASP, Y, MODEL, MSCORE, SRC = [], [], [], [], [], []
    stats = {}
    for shard in sorted(Path(a.raw).glob("*.parquet")):
        if not (shard.with_suffix(".parquet.ok")).exists():
            continue  # incomplete download
        mark = shard.with_suffix(".parquet.ok").read_text(encoding="utf-8").split()
        pf = pq.ParquetFile(shard)
        # a partial download (fetch_dataset.py --row-groups N) holds only its first N row groups
        t = pf.read_row_groups(list(range(int(mark[-1])))) if len(mark) >= 3 and mark[-2] == "partial" else pf.read()
        imgs, labels = t.column("image").to_pylist(), t.column("label").to_pylist()
        n = len(labels) if not a.limit else min(a.limit, len(labels))
        for i in range(n):
            name = HAGRID_CLASSES[labels[i]]
            s = stats.setdefault(name, {"images": 0, "kept": 0, "none": 0, "two": 0})
            s["images"] += 1
            bgr = cv2.imdecode(np.frombuffer(imgs[i]["bytes"], np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            h, w = bgr.shape[:2]
            res = rec.recognize(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            hands = res.hand_landmarks or []
            if len(hands) != 1:
                s["none" if not hands else "two"] += 1
                continue
            L.append([[p.x, p.y, p.z] for p in hands[0]])
            ASP.append(w / h)
            Y.append(name)
            g = (res.gestures or [[]])[0]
            MODEL.append(g[0].category_name if g else "")
            MSCORE.append(g[0].score if g else 0.0)
            SRC.append(f"{shard.stem}:{i}")
            s["kept"] += 1
        print(f"{shard.name} done", flush=True)
    np.savez_compressed(a.out, landmarks=np.asarray(L, np.float32), aspect=np.asarray(ASP, np.float32), label=np.asarray(Y),
                        model_label=np.asarray(MODEL), model_score=np.asarray(MSCORE, np.float32), source=np.asarray(SRC))
    for k, v in sorted(stats.items()):
        print(f"{k:14s} images {v['images']:4d}  kept {v['kept']:4d}  no hand {v['none']:3d}  two hands {v['two']:3d}")
    print("saved", a.out, len(Y))


if __name__ == "__main__":
    sys.exit(main())
