"""Export a few sample images per HaGRID class from the downloaded shards, to look at the data.

The images are stored *inside* the Parquet files (data/hagrid/raw/*.parquet, column "image");
this writes one contact sheet per class to data/hagrid/samples/<class>.jpg (git-ignored, local).
Usage:  python training/show_samples.py [--per-class 6] [--only three,peace]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_landmarks import HAGRID_CLASSES  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data/hagrid/raw"))
    ap.add_argument("--out", default=str(ROOT / "data/hagrid/samples"))
    ap.add_argument("--per-class", type=int, default=6)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    only = {s for s in a.only.split(",") if s}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sheets: dict[str, list[np.ndarray]] = {}
    for shard in sorted(Path(a.raw).glob("*.parquet")):
        if not shard.with_suffix(".parquet.ok").exists():
            continue
        t = pq.ParquetFile(shard).read_row_groups([0])  # the first row group is always present
        for img, lab in zip(t.column("image").to_pylist(), t.column("label").to_pylist()):
            name = HAGRID_CLASSES[lab]
            if (only and name not in only) or len(sheets.setdefault(name, [])) >= a.per_class:
                continue
            bgr = cv2.imdecode(np.frombuffer(img["bytes"], np.uint8), cv2.IMREAD_COLOR)
            if bgr is not None:
                sheets[name].append(cv2.resize(bgr, (200, int(200 * bgr.shape[0] / bgr.shape[1]))))
    for name, imgs in sorted(sheets.items()):
        if not imgs:
            continue
        h = max(i.shape[0] for i in imgs)
        row = np.hstack([cv2.copyMakeBorder(i, 0, h - i.shape[0], 0, 4, cv2.BORDER_CONSTANT) for i in imgs])
        cv2.imwrite(str(out / f"{name}.jpg"), row)
        print(f"{name}: {len(imgs)} images -> {out / (name + '.jpg')}")


if __name__ == "__main__":
    sys.exit(main())
