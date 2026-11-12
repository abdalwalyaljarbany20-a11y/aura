"""Localhost WebSocket server delivering engine results to the existing frontend.

* binds to 127.0.0.1 by default; rejects browser connections from other origins
* one client at a time (the camera has one consumer); others are refused
* the camera opens on the client's "start" and is released on "stop" or disconnect
* results are not queued: if the client falls behind, an unsent result is replaced by
  the newer one, with its gesture events carried over so no command is lost
"""

from __future__ import annotations

import asyncio
import logging
import threading
from http import HTTPStatus

from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from .engine import Engine
from .metrics import now_ms
from .protocol import (MAX_CLIENT_MESSAGE_BYTES, dumps, encode_preview, origin_allowed, parse_client_message)

log = logging.getLogger("aura_cv.server")


class Outbox:
    """Latest-result mailbox shared between the processing thread and the event loop."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._lock = threading.Lock()
        self._result: dict | None = None
        self._preview: bytes | None = None
        self._statuses: list[dict] = []
        self.event = asyncio.Event()
        self.replaced = 0

    def put_result(self, result: dict, preview: bytes | None) -> None:
        with self._lock:
            if self._result is not None:
                self.replaced += 1
                result["events"] = self._result.get("events", []) + result.get("events", [])
            self._result, self._preview = result, preview
        self._loop.call_soon_threadsafe(self.event.set)

    def put_status(self, status: dict) -> None:
        with self._lock:
            self._statuses.append(status)
        self._loop.call_soon_threadsafe(self.event.set)

    def take(self) -> tuple[list[dict], dict | None, bytes | None]:
        with self._lock:
            out = (self._statuses, self._result, self._preview)
            self._statuses, self._result, self._preview = [], None, None
            self.event.clear()
            return out

    def clear(self) -> None:
        self.take()


class EngineServer:
    def __init__(self, engine: Engine, config_errors: list[str], config_source: str) -> None:
        self.engine = engine
        self.config_errors = config_errors
        self.config_source = config_source
        self.client: ServerConnection | None = None
        self.outbox: Outbox | None = None

    async def run(self, host: str, port: int, ready: asyncio.Event | None = None,
                  stop: asyncio.Event | None = None) -> None:
        loop = asyncio.get_running_loop()
        self.outbox = Outbox(loop)
        self.engine.on_status = lambda s: self.outbox.put_status({"type": "status", **s})
        self.engine.on_result = self.outbox.put_result
        log.info("loading hand landmark model…")
        await asyncio.to_thread(self.engine.init_tracker)
        async with serve(self._handle, host, port, process_request=self._check_origin,
                         max_size=MAX_CLIENT_MESSAGE_BYTES, compression=None) as server:
            bound = ", ".join(f"{s.getsockname()[0]}:{s.getsockname()[1]}" for s in server.sockets)
            log.info("AURA CV engine listening on ws://%s (tracker %s)", bound, self.engine.tracker.status)
            self.port = server.sockets[0].getsockname()[1]
            if ready:
                ready.set()
            try:
                await (stop.wait() if stop else asyncio.Future())
            finally:
                await asyncio.to_thread(self.engine.shutdown)

    def _check_origin(self, connection: ServerConnection, request):
        origin = request.headers.get("Origin")
        if not origin_allowed(origin, self.engine.config["backend"]["allowedOrigins"]):
            log.warning("rejected connection from origin %s", origin)
            return connection.respond(HTTPStatus.FORBIDDEN, "origin not allowed\n")
        return None

    async def _handle(self, ws: ServerConnection) -> None:
        if self.client is not None:
            await ws.close(1013, "engine busy: another client is connected")
            return
        self.client = ws
        self.outbox.clear()
        log.info("client connected from %s", ws.remote_address)
        sender = asyncio.create_task(self._send_loop(ws))
        metrics = asyncio.create_task(self._metrics_loop(ws))
        for task in (sender, metrics):
            task.add_done_callback(lambda t, ws=ws: self._task_done(t, ws))
        try:
            await ws.send(dumps(self.engine.hello(self.config_errors, self.config_source)))
            async for raw in ws:
                msg, err = parse_client_message(raw)
                if err:
                    await ws.send(dumps({"type": "error", "message": err}))
                    continue
                await self._on_message(ws, msg)
        except ConnectionClosed:
            pass
        finally:
            sender.cancel()
            metrics.cancel()
            await asyncio.to_thread(self.engine.stop)  # release the camera when the client goes away
            self.client = None
            log.info("client disconnected; camera released")

    def _task_done(self, task: asyncio.Task, ws: ServerConnection) -> None:
        """A failing sender must not leave the client connected but starved."""
        if task.cancelled() or task.exception() is None or isinstance(task.exception(), ConnectionClosed):
            return
        log.error("send task failed; closing client connection", exc_info=task.exception())
        asyncio.ensure_future(ws.close(1011, "engine internal error"))

    async def _on_message(self, ws: ServerConnection, msg: dict) -> None:
        t = msg["type"]
        if t == "start":
            ok, err = await asyncio.to_thread(self.engine.start)
            if not ok:
                await ws.send(dumps({"type": "status", **self.engine.status(), "startError": err}))
        elif t == "stop":
            await asyncio.to_thread(self.engine.stop)
        elif t == "ping":
            await ws.send(dumps({"type": "pong", "t": msg["t"], "serverTime": round(now_ms(), 3)}))
        # hello: nothing to do beyond validation (version checked by the parser)

    async def _send_loop(self, ws: ServerConnection) -> None:
        while True:
            await self.outbox.event.wait()
            statuses, result, preview = self.outbox.take()
            for s in statuses:
                await ws.send(dumps(s))
            if result is not None:
                result["sentAt"] = round(now_ms(), 2)
                if result.get("capturedAt") is not None:
                    result["ageAtSendMs"] = round(result["sentAt"] - result["capturedAt"], 2)
                await ws.send(dumps(result))
                if preview is not None and result.get("frameId") is not None:
                    await ws.send(encode_preview(result["session"], result["frameId"], preview))

    async def _metrics_loop(self, ws: ServerConnection) -> None:
        while True:
            await asyncio.sleep(0.5)
            m = self.engine.metrics()
            if m is not None:
                m["resultsReplaced"] = self.outbox.replaced
                await ws.send(dumps({"type": "metrics", **m}))
