"""
Timeout Utilities.

Provides async timeout handling for generation requests
with proper cleanup and error responses.
"""

from __future__ import annotations

import asyncio
import os
from typing import Awaitable, TypeVar

T = TypeVar("T")


class TimeoutError(Exception):
    """Raised when an operation times out."""

    def __init__(self, message: str = "Operation timed out", seconds: float = 0) -> None:
        """Initialize the timeout error.

        Args:
            message: Error message.
            seconds: Timeout duration that was exceeded.
        """
        self.message = message
        self.seconds = seconds
        super().__init__(message)


async def with_timeout(
    coro: Awaitable[T],
    seconds: float,
    error_message: str | None = None,
) -> T:
    """Execute a coroutine with a timeout.

    Args:
        coro: The coroutine to execute.
        seconds: Maximum time to wait in seconds.
        error_message: Custom error message for timeout.

    Returns:
        The result of the coroutine.

    Raises:
        TimeoutError: If the operation times out.
    """
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except asyncio.TimeoutError:
        message = error_message or f"Operation timed out after {seconds} seconds"
        raise TimeoutError(message, seconds) from None


def get_generation_timeout() -> float:
    """Get the configured generation timeout.

    Returns:
        Timeout in seconds for generation requests.
    """
    return float(os.getenv("GENERATION_TIMEOUT_SECONDS", "120"))


def get_streaming_chunk_timeout() -> float:
    """Get the configured streaming chunk timeout.

    Returns:
        Timeout in seconds between streaming chunks.
    """
    return float(os.getenv("STREAMING_CHUNK_TIMEOUT_SECONDS", "30"))
