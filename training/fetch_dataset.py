"""Download a class-balanced subset of HaGRIDv2 (512 px) for training the hand-pose model.

Source: Hugging Face mirror `testdummyvt/hagRIDv2_512px` of HaGRIDv2 by Kapitanov et al.,
licence: https://github.com/hukenovs/hagrid/blob/master/license/en_us.pdf (a variant of
CC BY-SA 4.0). The images are only read locally to extract hand landmarks; they are never
uploaded and are not part of the repository (data/ is git-ignored).

The test split is stored sorted by label, about 550 rows per shard, so one shard holds about
550 images of a single class. SHARDS picks one (or more) shard inside each class block. The
label of every row is re-checked after download (extract_landmarks.py), so a shard that
straddles a class boundary is still labelled correctly.

Usage:  python training/fetch_dataset.py [--out data/hagrid/raw] [--only palm,fist] [--row-groups 2]
Resumable: partial files are continued with an HTTP Range request.
--row-groups N (slow networks): each shard holds 6 row groups of 100 images. Only the Parquet
footer and the first N row groups are downloaded (HTTP Range); the file is reassembled at its
full size with the unread row groups left as zeros, and the .ok marker records "partial N" so
extract_landmarks.py reads only those row groups. Full shards already downloaded are kept.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/testdummyvt/hagRIDv2_512px/resolve/main/data/test/test-{:05d}-of-00301.parquet"

# HaGRID class -> test-split shard indices (class k starts at row 5000k, or 5000k - 4500 after no_gesture).
SHARDS: dict[str, list[int]] = {
    "palm": [139], "stop": [185], "fist": [20], "one": [130], "point": [167],
    "peace": [148], "two_up": [276], "three": [212], "three2": [221], "three3": [230],
    "like": [84], "dislike": [11], "no_gesture": [118],
    # negatives (poses the app must not confuse with its commands)
    "four": [29], "ok": [121], "call": [2], "rock": [176], "grabbing": [38],
    "three_gun": [239], "little_finger": [93], "timeout": [270], "mute": [111],
}


def fetch(url: str, dest: Path, retries: int = 8) -> None:
    for attempt in range(retries):
        have = dest.stat().st_size if dest.exists() else 0
        headers = {"User-Agent": "aura-dataset-fetch"}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
                if have and r.status != 206:
                    have = 0  # server ignored the range: start over
                total = have + int(r.headers.get("Content-Length", 0))
                with open(dest, "ab" if have else "wb") as f:
                    while chunk := r.read(1 << 16):
                        f.write(chunk)
            if dest.stat().st_size >= total:
                return
        except urllib.error.HTTPError as e:
            if e.code == 416:  # range not satisfiable: already complete
                return
            print(f"  {dest.name}: HTTP {e.code}, retry {attempt + 1}", flush=True)
        except Exception as e:  # network drop: resume
            print(f"  {dest.name}: {type(e).__name__}, resume {attempt + 1}", flush=True)
        time.sleep(min(30, 2 ** attempt))
    raise SystemExit(f"failed: {url}")


def get_range(url: str, rng: str, retries: int = 8) -> tuple[bytes, int]:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "aura-dataset-fetch", "Range": f"bytes={rng}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                total = int(r.headers["Content-Range"].split("/")[-1])
                return r.read(), total
        except Exception as e:
            print(f"  range {rng}: {type(e).__name__}, retry {attempt + 1}", flush=True)
            time.sleep(min(30, 2 ** attempt))
    raise SystemExit(f"failed: {url} range {rng}")


def fetch_partial(url: str, dest: Path, groups: int) -> int:
    """Footer + first `groups` row groups; returns the number of row groups made readable."""
    import io

    import pyarrow.parquet as pq

    tail, total = get_range(url, "-8")
    flen = int.from_bytes(tail[:4], "little")
    footer, _ = get_range(url, f"{total - 8 - flen}-{total - 1}")
    meta = pq.read_metadata(io.BytesIO(b"PAR1" + bytes(total - 12 - flen) + footer))  # footer-only view
    groups = min(groups, meta.num_row_groups)
    end = 0
    for g in range(groups):
        rg = meta.row_group(g)
        for c in range(rg.num_columns):
            col = rg.column(c)
            start = col.dictionary_page_offset if col.has_dictionary_page and col.dictionary_page_offset else col.data_page_offset
            end = max(end, start + col.total_compressed_size)
    # an interrupted full download is a valid prefix: continue it up to `end` in 4 MB ranges
    while (have := (dest.stat().st_size if dest.exists() else 0)) < end:
        chunk, _ = get_range(url, f"{have}-{min(end, have + (4 << 20)) - 1}")
        with open(dest, "ab") as f:
            f.write(chunk)
    with open(dest, "r+b") as f:  # reassemble at full size: prefix, zeros, footer
        f.truncate(end)
        f.seek(total - 8 - flen)
        f.write(footer)
    return groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/hagrid/raw")
    ap.add_argument("--only", default="")
    ap.add_argument("--row-groups", type=int, default=0, help="download only the first N row groups (100 images each)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    only = {s for s in a.only.split(",") if s}
    for cls, shards in SHARDS.items():
        if only and cls not in only:
            continue
        for s in shards:
            dest = out / f"test-{s:05d}.parquet"
            done = out / f"test-{s:05d}.parquet.ok"
            if done.exists():
                continue
            t0 = time.time()
            if a.row_groups:
                n = fetch_partial(BASE.format(s), dest, a.row_groups)
                done.write_text(f"{cls} partial {n}")
            else:
                fetch(BASE.format(s), dest)
                done.write_text(cls)
            print(f"{cls}: shard {s} {dest.stat().st_size / 1e6:.1f} MB in {time.time() - t0:.0f} s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
