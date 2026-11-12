"""WebSocket server: handshake, origin policy, single client, start/stop, result+preview."""

import asyncio
import copy
import json

import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

from aura_cv.config import DEFAULT_CONFIG
from aura_cv.engine import Engine
from aura_cv.protocol import decode_preview
from aura_cv.server import EngineServer

from fixtures import POSES, FakeCapture, ScriptedBackend, blank_frame, make_hand, mp_result


async def start_server(tmp_path, cap):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    engine = Engine(cfg, tmp_path, backend=ScriptedBackend([mp_result(make_hand(POSES["OPEN_PALM"]))]),
                    capture_factory=lambda i, a: (cap, "FAKE"))
    server = EngineServer(engine, [], "defaults")
    ready, stop = asyncio.Event(), asyncio.Event()
    task = asyncio.create_task(server.run("127.0.0.1", 0, ready=ready, stop=stop))
    await asyncio.wait_for(ready.wait(), 10)
    return server, engine, stop, task, f"ws://127.0.0.1:{server.port}"


async def recv_until(ws, pred, timeout=5.0):
    async def loop():
        while True:
            m = await ws.recv()
            if pred(m):
                return m

    return await asyncio.wait_for(loop(), timeout)


def is_type(t):
    return lambda m: isinstance(m, str) and json.loads(m)["type"] == t


def test_server_end_to_end(tmp_path):
    async def scenario():
        cap = FakeCapture([blank_frame(640, 360)], delay_s=0.01)
        server, engine, stop, task, url = await start_server(tmp_path, cap)
        try:
            async with connect(url, origin="http://127.0.0.1:5173") as ws:
                hello = json.loads(await ws.recv())
                assert hello["type"] == "hello" and hello["protocol"] == 2
                assert hello["tracker"]["status"] == "ready" and hello["engine"]["name"] == "aura_cv"

                # second client is refused while the first is connected
                async with connect(url) as ws2:
                    with pytest.raises(ConnectionClosed) as exc:
                        await asyncio.wait_for(ws2.recv(), 5)
                    assert exc.value.rcvd.code == 1013

                await ws.send(json.dumps({"type": "start"}))
                live = await recv_until(ws, lambda m: isinstance(m, str) and json.loads(m).get("camera", {}).get("status") == "live")
                assert json.loads(live)["camera"]["info"]["width"] == 640

                result = json.loads(await recv_until(ws, lambda m: isinstance(m, str)
                                                     and json.loads(m)["type"] == "result"
                                                     and json.loads(m)["gesture"]["stable"] == "OPEN_PALM"))
                assert result["status"] == "DETECTED" and result["ageAtSendMs"] >= 0
                session, frame_id, jpeg = decode_preview(await recv_until(ws, lambda m: isinstance(m, bytes)))
                assert session == result["session"] == 1
                assert json.loads(live)["session"] == result["session"], "live status announces the results' session"
                assert frame_id >= result["frameId"] and jpeg[:2] == b"\xff\xd8"
                metrics = json.loads(await recv_until(ws, is_type("metrics")))
                assert metrics["processedFps"] is None or metrics["processedFps"] > 0

                await ws.send("{bad json")
                err = json.loads(await recv_until(ws, is_type("error")))
                assert "invalid JSON" in err["message"]
                await ws.send(json.dumps({"type": "exec", "cmd": "calc.exe"}))
                assert "unknown message type" in json.loads(await recv_until(ws, is_type("error")))["message"]

            # client gone -> camera released
            for _ in range(100):
                if cap.released:
                    break
                await asyncio.sleep(0.02)
            assert cap.released == 1 and engine.camera.status == "stopped"
        finally:
            stop.set()
            await task
        assert engine.tracker.status == "closed"

    asyncio.run(scenario())


def test_server_rejects_foreign_origin(tmp_path):
    async def scenario():
        server, engine, stop, task, url = await start_server(tmp_path, FakeCapture([blank_frame()]))
        try:
            with pytest.raises(InvalidStatus) as exc:
                async with connect(url, origin="https://evil.example"):
                    pass
            assert exc.value.response.status_code == 403
        finally:
            stop.set()
            await task

    asyncio.run(scenario())


def test_server_reports_camera_failure(tmp_path):
    async def scenario():
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        engine = Engine(cfg, tmp_path, backend=ScriptedBackend([]), capture_factory=lambda i, a: (None, None))
        server = EngineServer(engine, [], "defaults")
        ready, stop = asyncio.Event(), asyncio.Event()
        task = asyncio.create_task(server.run("127.0.0.1", 0, ready=ready, stop=stop))
        await ready.wait()
        try:
            async with connect(f"ws://127.0.0.1:{server.port}") as ws:
                await ws.recv()
                await ws.send(json.dumps({"type": "start"}))
                msg = json.loads(await recv_until(ws, lambda m: isinstance(m, str) and "startError" in json.loads(m)))
                assert msg["startError"]["code"] == "UNAVAILABLE" and msg["camera"]["status"] == "error"
        finally:
            stop.set()
            await task

    asyncio.run(scenario())
