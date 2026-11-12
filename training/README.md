# Training the AURA hand-pose network

A small neural network (MLP, 63 → 128 → 64 → 8, about 17 k parameters) that recognises hand poses from the 21 MediaPipe hand landmarks. It is trained here on a public dataset and runs locally in both engines.

| | |
|---|---|
| Why | The bundled GestureRecognizer has no class for **three fingers** (RUN) and answers `None` for many poses. This network adds that class, and a confident **OTHER** rejects poses the app does not use. |
| Input | `poseVector`: 21 landmarks relative to the wrist, divided by the palm size (63 values). The same function runs in training (`backend/aura_cv/posenet.py`) and in the browser (`src/perception/handPoseNet.js`), and the two are parity-tested. |
| Output | `OPEN_PALM, FIST, POINT, TWO_FINGERS, THREE_FINGERS, THUMB_UP, THUMB_DOWN, OTHER` (softmax) |
| Runtime file | `public/models/aura_handpose.json` (weights, input normalisation, dataset description and metrics) |
| Without the file | The app runs as before: canned model + rules. A missing or malformed file is reported in the HUD log and the Python `hello` message, never fatal. |

## Dataset

**HaGRIDv2** (Kapitanov et al., *HaGRID: HAnd Gesture Recognition Image Dataset*, WACV 2024; HaGRIDv2, arXiv 2412.01508):
- 1.08 M images, 33 gestures plus `no_gesture`.
- The official host (sbercloud) refuses connections from the development network. The Hugging Face mirror [`testdummyvt/hagRIDv2_512px`](https://huggingface.co/datasets/testdummyvt/hagRIDv2_512px) (512 px) is used instead.
- Its test split is stored sorted by label, about 550 rows per Parquet shard. `fetch_dataset.py` therefore downloads **one shard per class (≈550 images, ≈65 MB each, 22 shards, ≈1.4 GB)**, not the 128 GB dataset.
- **Licence:** a variant of CC BY-SA 4.0 ([text](https://github.com/hukenovs/hagrid/blob/master/license/en_us.pdf)). Check it before redistributing the trained weights.
- The images are stored only in `data/` (git-ignored) and are only read locally.

Class mapping (HaGRID → app):

| App class | HaGRID classes |
|---|---|
| OPEN_PALM | palm, stop |
| FIST | fist |
| POINT | one (index up) |
| TWO_FINGERS | peace, two_up |
| THREE_FINGERS | the HaGRID "three" variant with index + middle + ring up (verified, see below) |
| THUMB_UP / THUMB_DOWN | like / dislike |
| OTHER | no_gesture, four, ok, call, rock, grabbing, three_gun, little_finger, timeout, mute, point (pointing at the camera), the other "three" variants |

PINCH is not learned: the pinch distance decides it (the canned model has no pinch class either).

## Steps

```bash
.venv/Scripts/python.exe -m pip install pyarrow   # Parquet reader (training only)
.venv/Scripts/python.exe -I training/fetch_dataset.py        # resumable; ~1.4 GB (add --row-groups 2 on a slow network: 200 images per class)
.venv/Scripts/python.exe -I training/extract_landmarks.py    # runs the app's GestureRecognizer on every image (CPU)
.venv/Scripts/python.exe -I training/train_handpose.py --three three   # "three" = index + middle + ring (measured, see VALIDATION.md §14)
```

`extract_landmarks.py` keeps images with exactly one detected hand. HaGRID images sometimes show a second, relaxed hand, and the image-level label does not say which hand makes the gesture. It also stores the canned model's answer for every image, so the full app decision can be evaluated before and after the network.

`train_handpose.py`:
- **Training:** a stratified 80 / 20 split with a fixed seed. Augmentation: mirror, ±25° rotation and landmark jitter. Class-weighted cross-entropy, Adam with a cosine schedule, and the best epoch by validation loss.
- **Export check:** the exported file is read back through the runtime code and must reproduce the trained predictions.
- **Report:** `training/report.json`.

## Results

See the "Hand-pose network" section of [docs/VALIDATION.md](../docs/VALIDATION.md).
