"""
Cache module for LLM response caching.

Provides Redis-based caching for deterministic LLM responses
to reduce latency and compute costs.
"""

from app.cache.prompt_cache import PromptCache, get_prompt_cache
from app.cache.redis_client import (
    RedisClient,
    get_redis_client,
    initialize_redis,
    shutdown_redis,
)

__all__ = [
    "RedisClient",
    "get_redis_client",
    "initialize_redis",
    "shutdown_redis",
    "PromptCache",
    "get_prompt_cache",
]
