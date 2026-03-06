"""
Locust Load Tests for LLM Inference Service.

Provides realistic load testing scenarios for the OpenAI-compatible
API endpoints including completions, chat completions, and streaming.

Usage:
    locust -f tests/load/locustfile.py --host http://localhost:8000

Web UI will be available at http://localhost:8089
"""

from __future__ import annotations

import json
import random
import time
from typing import Any

from locust import HttpUser, between, task
from locust.clients import ResponseContextManager


# Sample prompts for testing
SAMPLE_PROMPTS = [
    "Explain the concept of machine learning in simple terms.",
    "What is the capital of France?",
    "Write a haiku about programming.",
    "List three benefits of exercise.",
    "Describe the water cycle.",
    "What is 2 + 2?",
    "Explain quantum computing briefly.",
    "What are the primary colors?",
    "Define artificial intelligence.",
    "How does photosynthesis work?",
]

# Sample chat conversations
SAMPLE_CONVERSATIONS = [
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ],
    [
        {"role": "system", "content": "You are a coding expert."},
        {"role": "user", "content": "What is a Python list comprehension?"},
    ],
    [
        {"role": "user", "content": "Explain what an API is."},
    ],
    [
        {"role": "system", "content": "You are a math tutor."},
        {"role": "user", "content": "What is the Pythagorean theorem?"},
    ],
    [
        {"role": "system", "content": "You are a friendly assistant."},
        {"role": "user", "content": "Tell me a fun fact."},
    ],
]


class LLMInferenceUser(HttpUser):
    """Simulates a user interacting with the LLM inference service.

    Attributes:
        wait_time: Random wait between 1-3 seconds between requests.
    """

    wait_time = between(1, 3)

    def on_start(self) -> None:
        """Run when a simulated user starts.

        Verifies the service is healthy before starting tests.
        """
        # Verify service is ready
        response = self.client.get("/ready")
        if response.status_code != 200:
            raise Exception(f"Service not ready: {response.text}")

    @task(3)
    def text_completion(self) -> None:
        """Test the /v1/completions endpoint.

        Weight of 3 means this runs 3x as often as weight-1 tasks.
        """
        prompt = random.choice(SAMPLE_PROMPTS)

        payload = {
            "model": "test-model",  # Will be overridden by server
            "prompt": prompt,
            "max_tokens": 50,
            "temperature": 0.7,
        }

        with self.client.post(
            "/v1/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            catch_response=True,
        ) as response:
            self._validate_completion_response(response, "completion")

    @task(5)
    def chat_completion(self) -> None:
        """Test the /v1/chat/completions endpoint.

        Weight of 5 means this is the most common request type.
        """
        messages = random.choice(SAMPLE_CONVERSATIONS)

        payload = {
            "model": "test-model",
            "messages": messages,
            "max_tokens": 50,
            "temperature": 0.7,
        }

        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            catch_response=True,
        ) as response:
            self._validate_chat_response(response)

    @task(1)
    def cached_completion(self) -> None:
        """Test cache behavior with deterministic requests.

        Uses temperature=0 to ensure cacheability.
        """
        # Use fixed prompt for cache hits
        payload = {
            "model": "test-model",
            "prompt": "What is 2 + 2?",
            "max_tokens": 20,
            "temperature": 0,  # Deterministic for caching
        }

        with self.client.post(
            "/v1/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            catch_response=True,
        ) as response:
            self._validate_completion_response(response, "cached_completion")

    @task(2)
    def streaming_completion(self) -> None:
        """Test streaming /v1/completions endpoint."""
        prompt = random.choice(SAMPLE_PROMPTS)

        payload = {
            "model": "test-model",
            "prompt": prompt,
            "max_tokens": 50,
            "temperature": 0.7,
            "stream": True,
        }

        with self.client.post(
            "/v1/completions",
            json=payload,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            stream=True,
            catch_response=True,
        ) as response:
            self._validate_streaming_response(response, "streaming_completion")

    @task(2)
    def streaming_chat(self) -> None:
        """Test streaming /v1/chat/completions endpoint."""
        messages = random.choice(SAMPLE_CONVERSATIONS)

        payload = {
            "model": "test-model",
            "messages": messages,
            "max_tokens": 50,
            "temperature": 0.7,
            "stream": True,
        }

        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            stream=True,
            catch_response=True,
        ) as response:
            self._validate_streaming_response(response, "streaming_chat")

    @task(1)
    def list_models(self) -> None:
        """Test the /v1/models endpoint."""
        with self.client.get("/v1/models", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "data" in data and len(data["data"]) > 0:
                        response.success()
                    else:
                        response.failure("No models returned")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Status {response.status_code}")

    @task(1)
    def health_check(self) -> None:
        """Test health endpoints."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")

    @task(1)
    def metrics_endpoint(self) -> None:
        """Test Prometheus metrics endpoint."""
        with self.client.get("/metrics", catch_response=True) as response:
            if response.status_code == 200:
                # Check for expected Prometheus format
                content = response.text
                if "llm_" in content or "# HELP" in content:
                    response.success()
                else:
                    response.failure("Invalid metrics format")
            else:
                response.failure(f"Metrics failed: {response.status_code}")

    def _validate_completion_response(
        self,
        response: ResponseContextManager,
        name: str,
    ) -> None:
        """Validate a completion response.

        Args:
            response: The Locust response context manager.
            name: Name for error reporting.
        """
        if response.status_code == 200:
            try:
                data = response.json()
                if "choices" in data and len(data["choices"]) > 0:
                    if data["choices"][0].get("text") is not None:
                        response.success()
                    else:
                        response.failure(f"{name}: No text in response")
                else:
                    response.failure(f"{name}: No choices in response")
            except json.JSONDecodeError:
                response.failure(f"{name}: Invalid JSON")
        elif response.status_code == 429:
            # Rate limited - mark as expected failure
            response.failure("Rate limited")
        elif response.status_code == 504:
            response.failure("Timeout")
        else:
            response.failure(f"{name}: Status {response.status_code}")

    def _validate_chat_response(
        self,
        response: ResponseContextManager,
    ) -> None:
        """Validate a chat completion response.

        Args:
            response: The Locust response context manager.
        """
        if response.status_code == 200:
            try:
                data = response.json()
                if "choices" in data and len(data["choices"]) > 0:
                    message = data["choices"][0].get("message", {})
                    if message.get("content") is not None:
                        response.success()
                    else:
                        response.failure("chat: No content in message")
                else:
                    response.failure("chat: No choices in response")
            except json.JSONDecodeError:
                response.failure("chat: Invalid JSON")
        elif response.status_code == 429:
            response.failure("Rate limited")
        elif response.status_code == 504:
            response.failure("Timeout")
        else:
            response.failure(f"chat: Status {response.status_code}")

    def _validate_streaming_response(
        self,
        response: ResponseContextManager,
        name: str,
    ) -> None:
        """Validate a streaming response.

        Args:
            response: The Locust response context manager.
            name: Name for error reporting.
        """
        if response.status_code == 200:
            try:
                # Read streaming content
                chunks = []
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        chunks.append(chunk.decode("utf-8"))

                content = "".join(chunks)

                # Check for SSE format and [DONE] marker
                if "data:" in content:
                    if "[DONE]" in content:
                        response.success()
                    else:
                        response.failure(f"{name}: No [DONE] marker")
                else:
                    response.failure(f"{name}: Invalid SSE format")
            except Exception as e:
                response.failure(f"{name}: {str(e)}")
        elif response.status_code == 429:
            response.failure("Rate limited")
        else:
            response.failure(f"{name}: Status {response.status_code}")


class HighLoadUser(HttpUser):
    """Simulates high-load scenario with minimal wait times.

    Use this user class for stress testing.
    """

    wait_time = between(0.1, 0.5)

    @task
    def rapid_completions(self) -> None:
        """Send rapid completion requests."""
        payload = {
            "model": "test-model",
            "prompt": "Hi",
            "max_tokens": 10,
            "temperature": 0.5,
        }

        self.client.post(
            "/v1/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
        )


class CacheTestUser(HttpUser):
    """Focused on testing cache hit/miss behavior.

    Sends identical requests to maximize cache hits.
    """

    wait_time = between(0.5, 1)

    @task(10)
    def cached_request(self) -> None:
        """Send cacheable requests (same params, temp=0)."""
        payload = {
            "model": "test-model",
            "prompt": "What is the speed of light?",
            "max_tokens": 30,
            "temperature": 0,  # Deterministic
        }

        with self.client.post(
            "/v1/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status {response.status_code}")

    @task(1)
    def non_cached_request(self) -> None:
        """Send non-cacheable request (random prompt)."""
        payload = {
            "model": "test-model",
            "prompt": f"Random: {random.random()}",
            "max_tokens": 20,
            "temperature": 0.8,  # Non-deterministic
        }

        self.client.post(
            "/v1/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
