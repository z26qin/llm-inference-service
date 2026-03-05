"""
Prompt Cache for LLM Responses.

Caches deterministic LLM responses (low temperature, n=1, non-streaming)
using SHA-256 hashed keys for efficient lookup.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field

from app.cache.redis_client import RedisClient, get_redis_client


@dataclass
class CacheConfig:
    """Configuration for prompt caching."""

    enabled: bool
    ttl_seconds: int
    temperature_threshold: float
    key_prefix: str

    @classmethod
    def from_env(cls) -> CacheConfig:
        """Load configuration from environment variables."""
        return cls(
            enabled=os.getenv("CACHE_ENABLED", "true").lower() == "true",
            ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "3600")),
            temperature_threshold=float(os.getenv("CACHE_TEMPERATURE_THRESHOLD", "0.01")),
            key_prefix="llm:cache:v1:",
        )


@dataclass
class CacheStats:
    """Statistics for cache operations."""

    hits: int = 0
    misses: int = 0
    stores: int = 0
    errors: int = 0

    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

    def to_dict(self) -> dict:
        """Convert to dictionary for metrics."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "stores": self.stores,
            "errors": self.errors,
            "hit_rate": round(self.hit_rate(), 4),
        }


@dataclass
class CacheEntry:
    """A cached LLM response."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str | None
    model: str


class PromptCache:
    """Cache for LLM prompt responses.

    Only caches deterministic requests where:
    - temperature <= threshold (default 0.01)
    - n == 1 (single completion)
    - stream == False (non-streaming)

    Uses SHA-256 hashing of (model, prompt, params) for cache keys.
    """

    def __init__(self, config: CacheConfig | None = None) -> None:
        """Initialize the prompt cache.

        Args:
            config: Cache configuration. If None, loads from environment.
        """
        self.config = config or CacheConfig.from_env()
        self.stats = CacheStats()

    def _generate_cache_key(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None,
        presence_penalty: float,
        frequency_penalty: float,
    ) -> str:
        """Generate a cache key from request parameters.

        Args:
            model: Model name.
            prompt: The prompt text.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            stop: Stop sequences.
            presence_penalty: Presence penalty.
            frequency_penalty: Frequency penalty.

        Returns:
            SHA-256 hash prefixed with key prefix.
        """
        # Create deterministic JSON representation
        key_data = json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stop": sorted(stop) if stop else None,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
            },
            sort_keys=True,
        )

        # Hash for compact key
        key_hash = hashlib.sha256(key_data.encode()).hexdigest()
        return f"{self.config.key_prefix}{key_hash}"

    def is_cacheable(
        self,
        temperature: float,
        n: int,
        stream: bool,
    ) -> bool:
        """Check if a request is cacheable.

        Args:
            temperature: Sampling temperature.
            n: Number of completions requested.
            stream: Whether streaming is enabled.

        Returns:
            True if the request can be cached.
        """
        if not self.config.enabled:
            return False

        # Only cache deterministic requests
        if temperature > self.config.temperature_threshold:
            return False

        # Only cache single completions
        if n != 1:
            return False

        # Don't cache streaming requests
        if stream:
            return False

        return True

    async def get(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None,
        presence_penalty: float,
        frequency_penalty: float,
    ) -> CacheEntry | None:
        """Get a cached response.

        Args:
            model: Model name.
            prompt: The prompt text.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            stop: Stop sequences.
            presence_penalty: Presence penalty.
            frequency_penalty: Frequency penalty.

        Returns:
            CacheEntry if found, None otherwise.
        """
        redis_client = await get_redis_client()
        if redis_client is None or not redis_client.is_connected():
            return None

        cache_key = self._generate_cache_key(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )

        try:
            cached_data = await redis_client.get(cache_key)
            if cached_data is None:
                self.stats.misses += 1
                return None

            # Parse cached entry
            data = json.loads(cached_data)
            self.stats.hits += 1

            return CacheEntry(
                text=data["text"],
                prompt_tokens=data["prompt_tokens"],
                completion_tokens=data["completion_tokens"],
                finish_reason=data.get("finish_reason"),
                model=data["model"],
            )
        except Exception:
            self.stats.errors += 1
            return None

    async def set(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None,
        presence_penalty: float,
        frequency_penalty: float,
        entry: CacheEntry,
    ) -> bool:
        """Store a response in cache.

        Args:
            model: Model name.
            prompt: The prompt text.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            stop: Stop sequences.
            presence_penalty: Presence penalty.
            frequency_penalty: Frequency penalty.
            entry: The cache entry to store.

        Returns:
            True if stored successfully.
        """
        redis_client = await get_redis_client()
        if redis_client is None or not redis_client.is_connected():
            return False

        cache_key = self._generate_cache_key(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )

        try:
            # Serialize entry
            data = json.dumps({
                "text": entry.text,
                "prompt_tokens": entry.prompt_tokens,
                "completion_tokens": entry.completion_tokens,
                "finish_reason": entry.finish_reason,
                "model": entry.model,
            })

            await redis_client.set(
                cache_key,
                data,
                ex=self.config.ttl_seconds,
            )
            self.stats.stores += 1
            return True
        except Exception:
            self.stats.errors += 1
            return False

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self.stats


# Global prompt cache instance
_cache_instance: PromptCache | None = None


def get_prompt_cache() -> PromptCache:
    """Get or create the global prompt cache instance.

    Returns:
        The singleton PromptCache instance.
    """
    global _cache_instance

    if _cache_instance is None:
        _cache_instance = PromptCache()

    return _cache_instance
