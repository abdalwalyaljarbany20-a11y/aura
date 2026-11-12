"""Measurement helpers. All timestamps are milliseconds from time.perf_counter()
(monotonic); see ``now_ms``."""

from __future__ import annotations

import math
import time
from collections import deque


def now_ms() -> float:
    return time.perf_counter() * 1000.0


class RateMeter:
    """Events per second over a sliding window (same definition as the JS RateMeter)."""

    def __init__(self, window_ms: float = 1000.0) -> None:
        self.window_ms = window_ms
        self._times: deque[float] = deque()
        self._start: float | None = None
        self.total = 0

    def tick(self, t: float, count: int = 1) -> None:
        if self._start is None:
            self._start = t
        for _ in range(count):
            self._times.append(t)
        self.total += count
        self._trim(t)

    def rate(self, now: float) -> float | None:
        if self._start is None:
            return None
        self._trim(now)
        span = min(self.window_ms, now - self._start)
        if span < 100:
            return None
        return len(self._times) / (span / 1000.0)

    def _trim(self, now: float) -> None:
        cutoff = now - self.window_ms
        while self._times and self._times[0] < cutoff:
            self._times.popleft()


class RollingStats:
    """Mean / p95 / last over the most recent ``size`` samples."""

    def __init__(self, size: int = 120) -> None:
        self._samples: deque[float] = deque(maxlen=size)
        self.count = 0

    def push(self, value: float) -> None:
        if value is None or not math.isfinite(value):
            return
        self._samples.append(float(value))
        self.count += 1

    def summary(self) -> dict:
        n = len(self._samples)
        if not n:
            return {"last": None, "mean": None, "p95": None, "samples": 0}
        ordered = sorted(self._samples)
        p95 = ordered[min(n - 1, max(0, math.ceil(0.95 * n) - 1))]
        return {"last": self._samples[-1], "mean": sum(self._samples) / n, "p95": p95, "samples": n}
