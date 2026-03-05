"""
Redis Client Manager.

Provides a singleton Redis connection with connection pooling,
health checks, and graceful shutdown support.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool


@dataclass
class RedisConfig:
    """Configuration for Redis connection.

    All settings are loaded from environment variables.
    """

    url: str
    pool_size: int
    socket_timeout: float
    socket_connect_timeout: float
    retry_on_timeout: bool

    @classmethod
    def from_env(cls) -> RedisConfig:
        """Load configuration from environment variables."""
        return cls(
            url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            pool_size=int(os.getenv("REDIS_POOL_SIZE", "10")),
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
            retry_on_timeout=True,
        )


class RedisClient:
    """Async Redis client with connection pooling.

    Provides methods for basic Redis operations with built-in
    health checks and graceful shutdown support.
    """

    def __init__(self, config: RedisConfig | None = None) -> None:
        """Initialize the Redis client.

        Args:
            config: Redis configuration. If None, loads from environment.
        """
        self.config = config or RedisConfig.from_env()
        self._pool: ConnectionPool | None = None
        self._client: redis.Redis | None = None
        self._connected = False

    async def connect(self) -> None:
        """Establish connection to Redis.

        Creates a connection pool and verifies connectivity.
        Should be called during application startup.
        """
        if self._connected:
            return

        self._pool = ConnectionPool.from_url(
            self.config.url,
            max_connections=self.config.pool_size,
            socket_timeout=self.config.socket_timeout,
            socket_connect_timeout=self.config.socket_connect_timeout,
            retry_on_timeout=self.config.retry_on_timeout,
        )

        self._client = redis.Redis(connection_pool=self._pool)

        # Verify connection
        await self._client.ping()
        self._connected = True

    async def disconnect(self) -> None:
        """Disconnect from Redis.

        Closes all connections in the pool.
        Should be called during application shutdown.
        """
        if self._client is not None:
            await self._client.aclose()
            self._client = None

        if self._pool is not None:
            await self._pool.disconnect()
            self._pool = None

        self._connected = False

    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected and self._client is not None

    async def health_check(self) -> bool:
        """Check Redis connectivity.

        Returns:
            True if Redis is healthy, False otherwise.
        """
        if not self.is_connected():
            return False

        try:
            assert self._client is not None
            await self._client.ping()
            return True
        except Exception:
            return False

    async def get(self, key: str) -> bytes | None:
        """Get a value from Redis.

        Args:
            key: The key to retrieve.

        Returns:
            The value as bytes, or None if not found.
        """
        if not self.is_connected():
            raise RuntimeError("Redis not connected")

        assert self._client is not None
        return await self._client.get(key)

    async def set(
        self,
        key: str,
        value: str | bytes,
        ex: int | None = None,
    ) -> bool:
        """Set a value in Redis.

        Args:
            key: The key to set.
            value: The value to store.
            ex: Optional expiration time in seconds.

        Returns:
            True if successful.
        """
        if not self.is_connected():
            raise RuntimeError("Redis not connected")

        assert self._client is not None
        await self._client.set(key, value, ex=ex)
        return True

    async def delete(self, key: str) -> int:
        """Delete a key from Redis.

        Args:
            key: The key to delete.

        Returns:
            Number of keys deleted.
        """
        if not self.is_connected():
            raise RuntimeError("Redis not connected")

        assert self._client is not None
        return await self._client.delete(key)

    async def exists(self, key: str) -> bool:
        """Check if a key exists.

        Args:
            key: The key to check.

        Returns:
            True if the key exists.
        """
        if not self.is_connected():
            raise RuntimeError("Redis not connected")

        assert self._client is not None
        return bool(await self._client.exists(key))

    async def zadd(
        self,
        name: str,
        mapping: dict[str, float],
    ) -> int:
        """Add members to a sorted set.

        Args:
            name: The sorted set name.
            mapping: Dictionary of member -> score.

        Returns:
            Number of elements added.
        """
        if not self.is_connected():
            raise RuntimeError("Redis not connected")

        assert self._client is not None
        return await self._client.zadd(name, mapping)

    async def zremrangebyscore(
        self,
        name: str,
        min_score: float,
        max_score: float,
    ) -> int:
        """Remove members from a sorted set by score range.

        Args:
            name: The sorted set name.
            min_score: Minimum score (inclusive).
            max_score: Maximum score (inclusive).

        Returns:
            Number of elements removed.
        """
        if not self.is_connected():
            raise RuntimeError("Redis not connected")

        assert self._client is not None
        return await self._client.zremrangebyscore(name, min_score, max_score)

    async def zcount(
        self,
        name: str,
        min_score: float,
        max_score: float,
    ) -> int:
        """Count members in a sorted set by score range.

        Args:
            name: The sorted set name.
            min_score: Minimum score (inclusive).
            max_score: Maximum score (inclusive).

        Returns:
            Number of elements in range.
        """
        if not self.is_connected():
            raise RuntimeError("Redis not connected")

        assert self._client is not None
        return await self._client.zcount(name, min_score, max_score)

    async def expire(self, name: str, time: int) -> bool:
        """Set expiration on a key.

        Args:
            name: The key name.
            time: Expiration time in seconds.

        Returns:
            True if timeout was set.
        """
        if not self.is_connected():
            raise RuntimeError("Redis not connected")

        assert self._client is not None
        return await self._client.expire(name, time)


# Global Redis client instance (singleton pattern)
_redis_instance: RedisClient | None = None
_redis_lock = asyncio.Lock()


async def get_redis_client() -> RedisClient | None:
    """Get the global Redis client instance.

    Returns:
        The singleton RedisClient instance, or None if not initialized.
    """
    return _redis_instance


async def initialize_redis() -> RedisClient | None:
    """Initialize and connect the global Redis client.

    Should be called during application startup.
    Returns None if Redis is disabled or connection fails.

    Returns:
        The initialized RedisClient, or None if unavailable.
    """
    global _redis_instance

    cache_enabled = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    rate_limit_use_redis = os.getenv("RATE_LIMIT_USE_REDIS", "true").lower() == "true"

    # Skip Redis if neither caching nor rate limiting needs it
    if not cache_enabled and not rate_limit_use_redis:
        return None

    async with _redis_lock:
        if _redis_instance is not None:
            return _redis_instance

        try:
            _redis_instance = RedisClient()
            await _redis_instance.connect()
            return _redis_instance
        except Exception as e:
            print(f"Warning: Failed to connect to Redis: {e}")
            print("Continuing without Redis (caching and distributed rate limiting disabled)")
            _redis_instance = None
            return None


async def shutdown_redis() -> None:
    """Shutdown the global Redis client.

    Should be called during application shutdown.
    """
    global _redis_instance

    async with _redis_lock:
        if _redis_instance is not None:
            await _redis_instance.disconnect()
            _redis_instance = None
