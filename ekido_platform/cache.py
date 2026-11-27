"""An optional Valkey cache (ticket #142). Postgres stays the source of truth: with no `VALKEY_URL`, or with Valkey down, every function
here answers "not cached" and the caller does what it did before. A cache problem may make a request slower, never wrong or failed.

Three uses:

- **who is calling**: an API key or a session token, looked up once and remembered for at most `AUTH_TTL` seconds. Any revoke, sign-out or
  erasure calls `forget_callers()`, which drops every remembered caller at once, so there is no list of keys to keep in step. If Valkey
  is unreachable at that moment, an old entry can be used again until it expires: **`AUTH_TTL` is the longest a revoked key can still work**.
- **rate limits** shared by every process of the Platform (the in-process limiter is the fallback).
- **webhooks already handled**: a shortcut only. The database's primary key still decides.

Secrets are never stored: a key or token is kept as a SHA-256 of itself.
"""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import time
from typing import Any, Callable

log = logging.getLogger(__name__)

AUTH_TTL = 30  # seconds
WEBHOOK_TTL = 7 * 24 * 3600
PAUSE_AFTER_ERROR = 5.0  # seconds without asking Valkey again, so a dead server costs one timeout and not one per request
_GENERATION = "ekido:auth:generation"
_COUNT = """
local now = redis.call('TIME')
local ms = now[1] * 1000 + math.floor(now[2] / 1000)
return redis.call('ZCOUNT', KEYS[1], ms - ARGV[1] * 1000, '+inf')
"""

# A sliding window, atomic on the server: drop old entries, count, add only if there is room. Returns 1 if allowed.
_ALLOW = """
local now = redis.call('TIME')
local ms = now[1] * 1000 + math.floor(now[2] / 1000)
redis.call('ZREMRANGEBYSCORE', KEYS[1], 0, ms - ARGV[1] * 1000)
if redis.call('ZCARD', KEYS[1]) >= tonumber(ARGV[2]) then return 0 end
redis.call('ZADD', KEYS[1], ms, ARGV[3])
redis.call('EXPIRE', KEYS[1], ARGV[1])
return 1
"""


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class Cache:
    """`client` is a redis-py client (Valkey speaks the same protocol), or None for "no cache"."""

    def __init__(self, client: Any = None, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._client, self._clock, self._paused_until = client, clock, 0.0

    @classmethod
    def from_url(cls, url: str) -> "Cache":
        if not url:
            return cls()
        import redis  # only needed when a URL is configured

        return cls(redis.Redis.from_url(url, socket_timeout=0.2, socket_connect_timeout=0.2, decode_responses=True))

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def _try(self, action: Callable[[Any], Any]) -> tuple[bool, Any]:
        """(True, result), or (False, None) if there is no cache or it failed. Nothing here ever raises to the caller."""
        if self._client is None or self._clock() < self._paused_until:
            return False, None
        try:
            return True, action(self._client)
        except Exception as exc:  # connection refused, timeout, out of memory, a script error: all mean "no cache this time"
            self._paused_until = self._clock() + PAUSE_AFTER_ERROR
            log.warning("cache unavailable (%s); using the database", type(exc).__name__)  # never the URL: it can hold a password
            return False, None

    # -- who is calling

    def _generation(self) -> str | None:
        ok, value = self._try(lambda c: c.get(_GENERATION))
        if not ok:
            return None
        if value:
            return value
        # Missing (first use, or evicted): a new random value, so nothing written before it can match again.
        ok, value = self._try(lambda c: (c.set(_GENERATION, secrets.token_hex(8), nx=True), c.get(_GENERATION))[1])
        return value if ok else None

    def caller(self, kind: str, secret: str, load: Callable[[], dict | None]) -> dict | None:
        """What `load()` returns for this secret, remembered for `AUTH_TTL` seconds. Only a found caller is remembered, never a refusal.

        The generation is read **before** the database: if a revoke happens in between, what is written is already out of date and unused."""
        generation = self._generation()
        if generation is None:
            return load()
        key = f"ekido:auth:{kind}:{digest(secret)}"
        ok, raw = self._try(lambda c: c.get(key))
        if ok and raw:
            try:
                entry = json.loads(raw)
                if entry["g"] == generation:
                    return entry["v"]
            except (ValueError, KeyError, TypeError):
                pass  # a damaged entry is the same as no entry
        value = load()
        if value is not None:
            self._try(lambda c: c.set(key, json.dumps({"g": generation, "v": value}), ex=AUTH_TTL))
        return value

    def forget_callers(self) -> None:
        """Call after the database commit of a revoke, a sign-out or an erasure."""
        self._paused_until = 0.0  # worth trying even if the last request failed
        self._try(lambda c: c.set(_GENERATION, secrets.token_hex(8)))

    # -- rate limits

    def allow(self, bucket: str, subject: str, limit: int, window: int) -> bool | None:
        """True or False from the shared window, or None when the cache cannot say (use the local limiter)."""
        key = f"ekido:rate:{bucket}:{digest(subject)}"
        ok, result = self._try(lambda c: c.eval(_ALLOW, 1, key, window, limit, secrets.token_hex(8)))
        return bool(result) if ok else None

    def count(self, bucket: str, subject: str, window: int) -> int | None:
        """How many entries are in the window, or None when the cache cannot say."""
        key = f"ekido:rate:{bucket}:{digest(subject)}"
        ok, result = self._try(lambda c: c.eval(_COUNT, 1, key, window))
        return int(result) if ok else None

    # -- webhooks

    def webhook_seen(self, event_key: str) -> bool:
        ok, found = self._try(lambda c: c.get(f"ekido:webhook:{digest(event_key)}"))
        return ok and found is not None

    def webhook_done(self, event_key: str) -> None:
        """Only after the database has committed the event."""
        self._try(lambda c: c.set(f"ekido:webhook:{digest(event_key)}", "1", ex=WEBHOOK_TTL))
