import json
import shutil
import subprocess
from pathlib import Path

import pytest

from aura_cv.config import DEFAULT_CONFIG, load_config, load_config_file
from aura_cv.protocol import (PREVIEW_MAGIC, decode_preview, dumps, encode_preview, origin_allowed,
                              parse_client_message)

REPO = Path(__file__).resolve().parents[2]


def test_defaults_without_overrides():
    cfg, errors = load_config(None)
    assert cfg == DEFAULT_CONFIG and errors == []


def test_valid_overrides_and_ignored_browser_sections():
    cfg, errors = load_config({
        "camera": {"width": 640, "facingMode": "user"},
        "gestures": {"swipe": {"minDistance": 3}, "definitions": {"FIST": {"holdMs": 900}}},
        "backend": {"cameraIndex": 1},
        "interaction": {"bindings": {}}, "scene": {}, "engine": {"mode": "python"},
    })
    assert errors == []
    assert cfg["camera"]["width"] == 640 and cfg["gestures"]["swipe"]["minDistance"] == 3
    assert cfg["gestures"]["definitions"]["FIST"]["holdMs"] == 900 and cfg["backend"]["cameraIndex"] == 1


def test_invalid_values_rejected_individually():
    cfg, errors = load_config({
        "camera": {"width": -1, "bogus": 1},
        "tracking": {"delegate": "TPU", "numHands": True},
        "gestures": {"minScore": 2, "definitions": {"WAVE": {}}},
        "backend": {"port": 70000, "allowedOrigins": ["ftp://x"]},
        "nope": {},
    })
    assert cfg["camera"]["width"] == 1280 and cfg["tracking"]["delegate"] == "GPU"
    for frag in ("camera.width", "camera.bogus", "tracking.delegate", "tracking.numHands", "gestures.minScore",
                 "definitions.WAVE", "backend.port", "backend.allowedOrigins", "nope"):
        assert any(frag in e for e in errors), (frag, errors)


def test_camera_fourcc_validation():
    assert load_config({"backend": {"cameraFourcc": "MJPG"}})[0]["backend"]["cameraFourcc"] == "MJPG"
    for bad in ("MJ", "MJPG1", 4, "M JP"):
        cfg, errors = load_config({"backend": {"cameraFourcc": bad}})
        assert cfg["backend"]["cameraFourcc"] == "auto" and any("cameraFourcc" in e for e in errors), bad


def test_config_file_missing_invalid_and_example(tmp_path):
    assert load_config_file(tmp_path / "missing.json")[2] == "defaults"
    bad = tmp_path / "bad.json"
    bad.write_text("{oops", encoding="utf-8")
    assert load_config_file(bad)[1]
    cfg, errors, _ = load_config_file(REPO / "public" / "aura.config.example.json")
    assert errors == []


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_shared_defaults_match_browser_engine():
    """The Python and JS engines must use identical gesture/camera/tracking defaults."""
    script = ("import('./src/config/defaultConfig.js').then(m => console.log(JSON.stringify("
              "{gestures: m.DEFAULT_CONFIG.gestures, camera: m.DEFAULT_CONFIG.camera, tracking: m.DEFAULT_CONFIG.tracking, face: m.DEFAULT_CONFIG.face})))")
    out = subprocess.run(["node", "-e", script], cwd=REPO, capture_output=True, text=True, check=True).stdout
    js = json.loads(out)
    assert js["gestures"] == DEFAULT_CONFIG["gestures"]
    for k, v in DEFAULT_CONFIG["face"].items():
        assert js["face"][k] == v, k
    for k, v in DEFAULT_CONFIG["camera"].items():
        assert js["camera"][k] == v, k
    for k, v in DEFAULT_CONFIG["tracking"].items():
        assert js["tracking"][k] == v, k


def test_client_message_validation():
    assert parse_client_message('{"type":"start"}')[0] == {"type": "start"}
    assert parse_client_message('{"type":"hello","protocol":2}')[1] is None
    assert "protocol" in parse_client_message('{"type":"hello","protocol":1}')[1]
    for raw, frag in [
        (b"\x00", "binary"), ("x" * 5000, "too large"), ("{bad", "invalid JSON"), ("[1]", "object"),
        ('{"type":"exec"}', "unknown"), ('{"type":"hello","protocol":9}', "protocol"), ('{"type":"ping"}', "numeric"),
    ]:
        msg, err = parse_client_message(raw)
        assert msg is None and frag in err


def test_origin_policy():
    allowed = DEFAULT_CONFIG["backend"]["allowedOrigins"]
    assert origin_allowed(None, allowed)  # non-browser local client
    assert origin_allowed("http://127.0.0.1:5173", allowed)
    assert origin_allowed("http://localhost:4173", allowed)
    assert not origin_allowed("https://evil.example", allowed)
    assert not origin_allowed("http://localhost.evil.example", allowed)
    assert not origin_allowed("http://127.0.0.1:5173.evil", allowed)


def test_preview_framing_and_json_sanitising():
    data = encode_preview(3, 42, b"\xff\xd8jpeg")
    assert data.startswith(PREVIEW_MAGIC) and decode_preview(data) == (3, 42, b"\xff\xd8jpeg")
    with pytest.raises(ValueError):
        decode_preview(b"nope")
    assert json.loads(dumps({"a": float("nan"), "b": [float("inf")]})) == {"a": None, "b": [None]}
