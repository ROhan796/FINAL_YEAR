"""Redis cache with graceful degradation.

If Redis is unreachable every operation is a no-op — the app stays correct and
just slower. That is the difference between "degraded" and "down".
"""

from __future__ import annotations

import hashlib
import json
import threading
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)

_client: Any = None
_available: bool | None = None
_lock = threading.Lock()
_NULL = object()


def _get_client() -> Any | None:
    global _client, _available
    if _available is False:
        return None
    if _client is not None:
        return _client
    with _lock:
        if _client is not None:
            return _client
        if not settings.cache_enabled:
            _available = False
            return None
        try:
            import redis

            _client = redis.Redis.from_url(
                settings.redis_url,
                socket_connect_timeout=1.0,
                socket_timeout=1.0,
                decode_responses=True,
            )
            _client.ping()
            _available = True
            log.info("cache_connected", url=settings.redis_url)
        except Exception as exc:  # noqa: BLE001 - any failure means "no cache"
            _available = False
            _client = None
            log.warning("cache_unavailable", error=str(exc))
    return _client


def cache_key(*parts: Any) -> str:
    raw = ":".join(str(p) for p in parts)
    return f"ck:{hashlib.sha256(raw.encode()).hexdigest()[:40]}"


def get_json(key: str, default: Any = None) -> Any:
    client = _get_client()
    if client is None:
        return default
    try:
        raw = client.get(key)
    except Exception:  # noqa: BLE001
        return default
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return default


def set_json(key: str, value: Any, ttl: int) -> bool:
    client = _get_client()
    if client is None:
        return False
    try:
        client.setex(key, ttl, json.dumps(value, default=str))
        return True
    except Exception:  # noqa: BLE001
        return False


def delete(*keys: str) -> None:
    client = _get_client()
    if client is None or not keys:
        return
    try:
        client.delete(*keys)
    except Exception:  # noqa: BLE001
        pass


def invalidate_pattern(prefix: str) -> None:
    """Best-effort sweep. Used to bust feed keys after a write."""
    client = _get_client()
    if client is None:
        return
    try:
        cursor = 0
        while True:
            cursor, keys = client.scan(cursor=cursor, match=f"{prefix}*", count=200)
            if keys:
                client.delete(*keys)
            if cursor == 0:
                break
    except Exception:  # noqa: BLE001
        pass


def get_or_set(key: str, ttl: int, producer: Any) -> Any:
    """Cache-aside with a per-key lock so concurrent misses do one fetch, not N."""
    cached = get_json(key, _NULL)
    if cached is not _NULL:
        return cached

    client = _get_client()
    lock_key = f"{key}:lock"
    if client is not None:
        try:
            if not client.set(lock_key, "1", nx=True, ex=5):
                # Another worker is computing it — wait briefly, then fall through.
                import time

                for _ in range(10):
                    time.sleep(0.1)
                    cached = get_json(key, _NULL)
                    if cached is not _NULL:
                        return cached
        except Exception:  # noqa: BLE001
            pass

    try:
        value = producer()
    finally:
        if client is not None:
            try:
                client.delete(lock_key)
            except Exception:  # noqa: BLE001
                pass

    set_json(key, value, ttl)
    return value


def reset_client() -> None:
    """Test helper — force a reconnect on the next call."""
    global _client, _available
    with _lock:
        _client = None
        _available = None
