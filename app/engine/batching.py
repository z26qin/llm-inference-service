"""
Intelligent Batching Module.

Provides a priority queue system for managing inference requests with
configurable batch sizes, timeouts, and client tier priorities.

Note: vLLM's AsyncLLMEngine handles continuous batching internally.
This module provides additional queue management for rate limiting,
prioritization, and request lifecycle management.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Awaitable, Callable
from uuid import uuid4

from app.engine.vllm_engine import GenerationOutput, GenerationRequest


class ClientTier(IntEnum):
    """Client tiers for request prioritization.

    Higher values indicate higher priority (processed first).
    """

    FREE = 0
    STANDARD = 1
    PREMIUM = 2
    ENTERPRISE = 3


@dataclass
class BatchConfig:
    """Configuration for batch processing.

    All settings loaded from environment variables.
    """

    max_batch_size: int
    max_wait_time_ms: float
    request_timeout_ms: float
    max_queue_size: int

    @classmethod
    def from_env(cls) -> BatchConfig:
        """Load configuration from environment variables."""
        return cls(
            max_batch_size=int(os.getenv("MAX_BATCH_SIZE", "32")),
            max_wait_time_ms=float(os.getenv("MAX_WAIT_TIME_MS", "100")),
            request_timeout_ms=float(os.getenv("REQUEST_TIMEOUT_MS", "30000")),
            max_queue_size=int(os.getenv("MAX_QUEUE_SIZE", "1000")),
        )


@dataclass(order=True)
class PrioritizedRequest:
    """A request with priority for queue ordering.

    Uses negative priority for max-heap behavior (higher priority first).
    """

    priority: int  # Negative of ClientTier for max-heap
    timestamp: float  # For FIFO within same priority
    request_id: str = field(compare=False)
    data: GenerationRequest = field(compare=False)
    future: asyncio.Future[GenerationOutput] = field(compare=False, repr=False)

    @classmethod
    def create(
        cls,
        data: GenerationRequest,
        tier: ClientTier = ClientTier.STANDARD,
        request_id: str | None = None,
    ) -> PrioritizedRequest:
        """Create a new prioritized request.

        Args:
            data: The request data.
            tier: Client tier for prioritization.
            request_id: Optional request ID (generated if not provided).

        Returns:
            A new PrioritizedRequest instance.
        """
        return cls(
            priority=-tier,  # Negative for max-heap (higher tier = lower value = first)
            timestamp=time.monotonic(),
            request_id=request_id or str(uuid4()),
            data=data,
            future=asyncio.get_event_loop().create_future(),
        )


class RequestQueue:
    """Priority queue for managing inference requests.

    Provides priority-based ordering with FIFO ordering within
    the same priority level. Supports timeout and cancellation.
    """

    def __init__(self, config: BatchConfig | None = None) -> None:
        """Initialize the request queue.

        Args:
            config: Queue configuration. Loads from env if not provided.
        """
        self.config = config or BatchConfig.from_env()
        self._queue: asyncio.PriorityQueue[PrioritizedRequest] = asyncio.PriorityQueue(
            maxsize=self.config.max_queue_size
        )
        self._pending_requests: dict[str, PrioritizedRequest] = {}
        self._lock = asyncio.Lock()

    @property
    def size(self) -> int:
        """Current number of requests in queue."""
        return self._queue.qsize()

    @property
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()

    @property
    def is_full(self) -> bool:
        """Check if queue is at capacity."""
        return self._queue.full()

    async def enqueue(
        self,
        data: GenerationRequest,
        tier: ClientTier = ClientTier.STANDARD,
        request_id: str | None = None,
        timeout: float | None = None,
    ) -> GenerationOutput:
        """Add a request to the queue and wait for result.

        Args:
            data: The request data.
            tier: Client tier for prioritization.
            request_id: Optional request ID.
            timeout: Optional timeout in seconds (uses config default if None).

        Returns:
            The result from processing the request.

        Raises:
            asyncio.TimeoutError: If request times out.
            asyncio.QueueFull: If queue is at capacity.
            asyncio.CancelledError: If request is cancelled.
        """
        request = PrioritizedRequest.create(data, tier, request_id)

        async with self._lock:
            if self._queue.full():
                raise asyncio.QueueFull("Request queue at capacity")
            await self._queue.put(request)
            self._pending_requests[request.request_id] = request

        timeout_sec = timeout or (self.config.request_timeout_ms / 1000)

        try:
            return await asyncio.wait_for(request.future, timeout=timeout_sec)
        except asyncio.TimeoutError:
            await self.cancel(request.request_id)
            raise
        finally:
            async with self._lock:
                self._pending_requests.pop(request.request_id, None)

    async def dequeue(self, timeout: float | None = None) -> PrioritizedRequest | None:
        """Remove and return the highest priority request.

        Args:
            timeout: Optional timeout in seconds.

        Returns:
            The highest priority request, or None if timeout.
        """
        try:
            if timeout is not None:
                return await asyncio.wait_for(self._queue.get(), timeout=timeout)
            return await self._queue.get()
        except asyncio.TimeoutError:
            return None

    async def dequeue_batch(
        self,
        max_size: int | None = None,
        max_wait_ms: float | None = None,
    ) -> list[PrioritizedRequest]:
        """Dequeue a batch of requests.

        Waits up to max_wait_ms for the first request, then collects
        up to max_size requests without additional waiting.

        Args:
            max_size: Maximum batch size (uses config default if None).
            max_wait_ms: Maximum wait time in ms (uses config default if None).

        Returns:
            List of dequeued requests (may be empty if timeout).
        """
        max_size = max_size or self.config.max_batch_size
        max_wait_ms = max_wait_ms or self.config.max_wait_time_ms

        batch: list[PrioritizedRequest] = []

        # Wait for first request with timeout
        first = await self.dequeue(timeout=max_wait_ms / 1000)
        if first is None:
            return batch

        batch.append(first)

        # Collect additional requests without waiting
        while len(batch) < max_size:
            try:
                request = self._queue.get_nowait()
                batch.append(request)
            except asyncio.QueueEmpty:
                break

        return batch

    async def cancel(self, request_id: str) -> bool:
        """Cancel a pending request.

        Args:
            request_id: ID of the request to cancel.

        Returns:
            True if request was found and cancelled.
        """
        async with self._lock:
            request = self._pending_requests.get(request_id)
            if request is None:
                return False

            if not request.future.done():
                request.future.cancel()

            self._pending_requests.pop(request_id, None)
            return True

    def complete(self, request_id: str, result: GenerationOutput) -> bool:
        """Complete a request with a result.

        Args:
            request_id: ID of the request to complete.
            result: The result to set.

        Returns:
            True if request was found and completed.
        """
        request = self._pending_requests.get(request_id)
        if request is None:
            return False

        if not request.future.done():
            request.future.set_result(result)
        return True

    def fail(self, request_id: str, exception: Exception) -> bool:
        """Fail a request with an exception.

        Args:
            request_id: ID of the request to fail.
            exception: The exception to set.

        Returns:
            True if request was found and failed.
        """
        request = self._pending_requests.get(request_id)
        if request is None:
            return False

        if not request.future.done():
            request.future.set_exception(exception)
        return True

    async def clear(self) -> int:
        """Clear all pending requests.

        Cancels all pending request futures.

        Returns:
            Number of requests cleared.
        """
        async with self._lock:
            count = len(self._pending_requests)
            for request in self._pending_requests.values():
                if not request.future.done():
                    request.future.cancel()
            self._pending_requests.clear()

            # Drain the queue
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

            return count


class BatchProcessor:
    """Processes batches of requests from a queue.

    Runs as a background task, collecting batches and dispatching
    them to a processor function.
    """

    def __init__(
        self,
        queue: RequestQueue,
        processor: Callable[[GenerationRequest, str], Awaitable[GenerationOutput]],
        config: BatchConfig | None = None,
    ) -> None:
        """Initialize the batch processor.

        Args:
            queue: The request queue to process from.
            processor: Async function to process individual requests.
            config: Batch configuration.
        """
        self.queue = queue
        self.processor = processor
        self.config = config or BatchConfig.from_env()
        self._running = False
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the batch processor background task."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._process_loop())

    async def stop(self) -> None:
        """Stop the batch processor gracefully."""
        self._running = False

        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _process_loop(self) -> None:
        """Main processing loop."""
        while self._running:
            try:
                batch = await self.queue.dequeue_batch()
                if not batch:
                    continue

                # Process each request in the batch concurrently
                tasks = [self._process_request(req) for req in batch]
                await asyncio.gather(*tasks, return_exceptions=True)

            except asyncio.CancelledError:
                break
            except Exception:
                # Log error but continue processing
                continue

    async def _process_request(self, request: PrioritizedRequest) -> None:
        """Process a single request.

        Args:
            request: The request to process.
        """
        try:
            result = await self.processor(request.data, request.request_id)
            self.queue.complete(request.request_id, result)
        except Exception as e:
            self.queue.fail(request.request_id, e)
