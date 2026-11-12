"""Wire protocol between the Python engine and the browser (version 2).

v2 (Phase 2): every camera session has an integer ``session`` id (increments on each
``start``). Results and previews carry it, so nothing from an earlier session can be
paired with, or override, the current one. Preview header: "AVF2" + session + frameId.

Transport: one WebSocket on 127.0.0.1. Text frames carry JSON; binary frames carry
preview images.

Engine -> browser
  hello    {type, protocol, engine{name, version, python, mediapipe, opencv, numpy},
            tracker{status, delegate, error}, camera{status, error, info}, config{...}, configErrors[]}
  status   {type, camera{status, error, info}, tracker{...}, session}   on every change
  result   see pipeline.py                                               one per processed frame
  metrics  {type, cameraFps, processedFps, inferenceMs{}, totalMs{}, droppedFrames, ...}  2 Hz
  pong     {type, t, serverTime}
  error    {type, message}
  preview  binary: b"AVF2" + uint32 BE session + uint32 BE frameId + JPEG bytes (same ids as its result)

Browser -> engine (validated; anything else is answered with an error)
  {type: "hello", client: str, protocol: 2}
  {type: "start"}  open the camera and start processing
  {type: "stop"}   stop processing and release the camera
  {type: "ping", t: number}
"""

from __future__ import annotations

import json
import math
import struct
from typing import Any

PROTOCOL_VERSION = 2
PREVIEW_MAGIC = b"AVF2"
CLIENT_TYPES = {"hello", "start", "stop", "ping"}
MAX_CLIENT_MESSAGE_BYTES = 4096


def encode_preview(session: int, frame_id: int, jpeg: bytes) -> bytes:
    return PREVIEW_MAGIC + struct.pack(">II", session & 0xFFFFFFFF, frame_id & 0xFFFFFFFF) + jpeg


def decode_preview(data: bytes) -> tuple[int, int, bytes]:
    if len(data) < 12 or data[:4] != PREVIEW_MAGIC:
        raise ValueError("not a preview frame")
    session, frame_id = struct.unpack(">II", data[4:12])
    return session, frame_id, data[12:]


def _sanitize(v: Any) -> Any:
    """Replace NaN/inf (not valid JSON) with None, recursively."""
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        return {k: _sanitize(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_sanitize(x) for x in v]
    return v


def dumps(msg: dict) -> str:
    return json.dumps(_sanitize(msg), separators=(",", ":"), allow_nan=False)


def parse_client_message(raw: str | bytes) -> tuple[dict | None, str | None]:
    """Validate an incoming message. Returns (message, error)."""
    if isinstance(raw, bytes):
        return None, "binary messages are not accepted"
    if len(raw) > MAX_CLIENT_MESSAGE_BYTES:
        return None, "message too large"
    try:
        msg = json.loads(raw)
    except ValueError:
        return None, "invalid JSON"
    if not isinstance(msg, dict):
        return None, "message must be a JSON object"
    t = msg.get("type")
    if t not in CLIENT_TYPES:
        return None, f"unknown message type {t!r}"
    if t == "hello" and msg.get("protocol") != PROTOCOL_VERSION:
        return None, f"unsupported protocol {msg.get('protocol')!r} (engine speaks {PROTOCOL_VERSION})"
    if t == "ping" and not (isinstance(msg.get("t"), (int, float)) and not isinstance(msg.get("t"), bool)):
        return None, "ping requires a numeric t"
    return msg, None


def origin_allowed(origin: str | None, allowed: list[str]) -> bool:
    """Allow non-browser clients (no Origin header) and configured scheme+host on any port.

    A browser always sends Origin, so this blocks other websites from opening the
    camera stream through the user's browser.
    """
    if origin is None:
        return True
    for a in allowed:
        if origin == a or (origin.startswith(a + ":") and origin[len(a) + 1:].isdigit()):
            return True
    return False
