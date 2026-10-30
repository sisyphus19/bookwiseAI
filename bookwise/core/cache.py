"""Tiny Redis JSON cache. Every operation degrades to a no-op if Redis is down (caching is an optimisation)."""
import hashlib
import json
from typing import Any

from shared.config import settings
from shared.logging import get_logger

logger = get_logger("cache")
_client = None
_failed = False


def _redis():
    global _client, _failed
    if _client is not None or _failed:
        return _client
    try:
        import redis
        c = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True, socket_timeout=1.0)
        c.ping()
        _client = c
    except Exception as e:
        _failed = True
        logger.warning(f"Redis cache disabled ({e}).")
    return _client


def make_key(prefix: str, **parts: Any) -> str:
    raw = json.dumps(parts, sort_keys=True, default=str)
    return f"{prefix}:{hashlib.sha256(raw.encode()).hexdigest()[:32]}"


def get_json(key: str) -> Any | None:
    c = _redis()
    if c is None:
        return None
    try:
        raw = c.get(key)
        return json.loads(raw) if raw else None
    except Exception as e:
        logger.warning(f"cache get failed: {e}")
        return None


def set_json(key: str, value: Any, ttl: int | None = None) -> None:
    c = _redis()
    if c is None:
        return
    try:
        c.setex(key, ttl or settings.CACHE_DEFAULT_TTL, json.dumps(value))
    except Exception as e:
        logger.warning(f"cache set failed: {e}")
