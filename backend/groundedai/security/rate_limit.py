"""In-memory token-bucket rate limiting for expensive endpoints.

Limits are per process. With several workers or replicas, put a shared limiter
(reverse proxy or gateway) in front; see docs/SECURITY.md.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from groundedai.core.errors import RateLimited

MAX_BUCKETS = 10_000


@dataclass(frozen=True, slots=True)
class Limit:
    capacity: int
    period_s: float

    @property
    def refill_per_s(self) -> float:
        return self.capacity / self.period_s


class TokenBucketLimiter:
    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._buckets: dict[tuple[str, str], tuple[float, float]] = {}

    def check(self, scope: str, key: str, limit: Limit) -> None:
        """Consume one token or raise :class:`RateLimited`."""
        now = self._clock()
        bucket_key = (scope, key)
        with self._lock:
            tokens, updated = self._buckets.get(bucket_key, (float(limit.capacity), now))
            tokens = min(float(limit.capacity), tokens + (now - updated) * limit.refill_per_s)
            if tokens < 1.0:
                retry_after = math.ceil((1.0 - tokens) / limit.refill_per_s)
                self._buckets[bucket_key] = (tokens, now)
                raise RateLimited(retry_after_s=max(1, retry_after))
            self._buckets[bucket_key] = (tokens - 1.0, now)
            if len(self._buckets) > MAX_BUCKETS:
                self._evict_full_buckets(now, limit)

    def _evict_full_buckets(self, now: float, limit: Limit) -> None:
        # Drop the oldest half; a removed bucket simply starts full again.
        ordered = sorted(self._buckets.items(), key=lambda item: item[1][1])
        for key, _ in ordered[: len(ordered) // 2]:
            del self._buckets[key]
