"""
vLLM Engine Wrapper.

Provides an async interface to vLLM's inference engine with support for
streaming and non-streaming completions, automatic batching, and
generation parameter management.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, AsyncIterator

from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs

if TYPE_CHECKING:
    from vllm.outputs import RequestOutput


@dataclass
class EngineConfig:
    """Configuration for the vLLM engine.

    All settings are loaded from environment variables following 12-factor app principles.
    """

    model_name: str
    tensor_parallel_size: int
    gpu_memory_utilization: float
    max_model_len: int | None
    trust_remote_code: bool
    dtype: str
    enforce_eager: bool
    max_num_seqs: int
    max_num_batched_tokens: int | None

    @classmethod
    def from_env(cls) -> EngineConfig:
        """Load configuration from environment variables."""
        return cls(
            model_name=os.getenv("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
            tensor_parallel_size=int(os.getenv("TENSOR_PARALLEL_SIZE", "1")),
            gpu_memory_utilization=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.90")),
            max_model_len=int(max_len) if (max_len := os.getenv("MAX_MODEL_LEN")) else None,
            trust_remote_code=os.getenv("TRUST_REMOTE_CODE", "true").lower() == "true",
            dtype=os.getenv("DTYPE", "auto"),
            enforce_eager=os.getenv("ENFORCE_EAGER", "false").lower() == "true",
            max_num_seqs=int(os.getenv("MAX_NUM_SEQS", "256")),
            max_num_batched_tokens=int(tokens)
            if (tokens := os.getenv("MAX_NUM_BATCHED_TOKENS"))
            else None,
        )


@dataclass
class GenerationRequest:
    """Request for text generation."""

    request_id: str
    prompt: str
    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    stop: list[str] | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    n: int = 1
    logprobs: int | None = None
    echo: bool = False
    best_of: int = 1

    def to_sampling_params(self) -> SamplingParams:
        """Convert to vLLM SamplingParams."""
        return SamplingParams(
            n=self.n,
            best_of=max(self.best_of, self.n),
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stop=self.stop,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            logprobs=self.logprobs,
            include_stop_str_in_output=False,
        )


@dataclass
class GenerationOutput:
    """Output from text generation."""

    request_id: str
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str | None
    logprobs: dict | None = None


class VLLMEngine:
    """Async wrapper for vLLM inference engine.

    Provides methods for both streaming and non-streaming completions,
    with automatic batching handled by vLLM's continuous batching engine.
    """

    def __init__(self, config: EngineConfig | None = None) -> None:
        """Initialize the engine with configuration.

        Args:
            config: Engine configuration. If None, loads from environment.
        """
        self.config = config or EngineConfig.from_env()
        self._engine: AsyncLLMEngine | None = None
        self._started = False

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.config.model_name

    async def start(self) -> None:
        """Start the vLLM engine.

        This initializes the model and prepares it for inference.
        Should be called during application startup.
        """
        if self._started:
            return

        engine_args = AsyncEngineArgs(
            model=self.config.model_name,
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_model_len=self.config.max_model_len,
            trust_remote_code=self.config.trust_remote_code,
            dtype=self.config.dtype,
            enforce_eager=self.config.enforce_eager,
            max_num_seqs=self.config.max_num_seqs,
            max_num_batched_tokens=self.config.max_num_batched_tokens,
            disable_log_stats=False,
        )

        self._engine = AsyncLLMEngine.from_engine_args(engine_args)
        self._started = True

    async def stop(self) -> None:
        """Stop the engine and release resources.

        Should be called during application shutdown for graceful cleanup.
        """
        if self._engine is not None:
            # vLLM doesn't have explicit shutdown, but we clear reference
            self._engine = None
        self._started = False

    def is_ready(self) -> bool:
        """Check if the engine is ready to serve requests."""
        return self._started and self._engine is not None

    async def generate(self, request: GenerationRequest) -> GenerationOutput:
        """Generate a completion for a single request.

        Args:
            request: The generation request.

        Returns:
            GenerationOutput with the completed text.

        Raises:
            RuntimeError: If engine is not started.
        """
        if not self.is_ready():
            raise RuntimeError("Engine not started. Call start() first.")

        assert self._engine is not None

        sampling_params = request.to_sampling_params()
        prompt = request.prompt

        # Add prompt to echo if requested
        prefix = prompt if request.echo else ""

        # Generate using vLLM's async engine
        final_output: RequestOutput | None = None
        async for output in self._engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request.request_id,
        ):
            final_output = output

        if final_output is None:
            raise RuntimeError("No output generated")

        # Extract the first completion (we support n=1 for now in non-streaming)
        completion_output = final_output.outputs[0]

        return GenerationOutput(
            request_id=request.request_id,
            text=prefix + completion_output.text,
            prompt_tokens=len(final_output.prompt_token_ids),
            completion_tokens=len(completion_output.token_ids),
            finish_reason=completion_output.finish_reason,
            logprobs=self._extract_logprobs(completion_output) if request.logprobs else None,
        )

    async def generate_stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[GenerationOutput]:
        """Generate a completion with streaming output.

        Yields partial outputs as they are generated.

        Args:
            request: The generation request.

        Yields:
            GenerationOutput with partial text at each step.

        Raises:
            RuntimeError: If engine is not started.
        """
        if not self.is_ready():
            raise RuntimeError("Engine not started. Call start() first.")

        assert self._engine is not None

        sampling_params = request.to_sampling_params()
        prompt = request.prompt

        # Track previous output for delta calculation
        previous_text = ""
        prompt_tokens = 0

        # Echo prompt first if requested
        if request.echo:
            yield GenerationOutput(
                request_id=request.request_id,
                text=prompt,
                prompt_tokens=0,
                completion_tokens=0,
                finish_reason=None,
            )

        async for output in self._engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request.request_id,
        ):
            prompt_tokens = len(output.prompt_token_ids)
            completion_output = output.outputs[0]
            current_text = completion_output.text

            # Calculate delta (new text since last yield)
            delta_text = current_text[len(previous_text) :]
            previous_text = current_text

            if delta_text or completion_output.finish_reason:
                yield GenerationOutput(
                    request_id=request.request_id,
                    text=delta_text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=len(completion_output.token_ids),
                    finish_reason=completion_output.finish_reason,
                    logprobs=self._extract_logprobs(completion_output)
                    if request.logprobs
                    else None,
                )

    async def abort_request(self, request_id: str) -> None:
        """Abort a running generation request.

        Args:
            request_id: The ID of the request to abort.
        """
        if self._engine is not None:
            await self._engine.abort(request_id)

    def _extract_logprobs(self, output: object) -> dict | None:
        """Extract log probabilities from completion output.

        Args:
            output: The completion output from vLLM.

        Returns:
            Dictionary of logprobs or None if not available.
        """
        if not hasattr(output, "logprobs") or output.logprobs is None:  # type: ignore
            return None

        # Convert vLLM logprobs format to OpenAI format
        logprobs_list = []
        for logprob_dict in output.logprobs:  # type: ignore
            if logprob_dict:
                top_logprobs = {
                    token: float(lp.logprob)
                    for token, lp in logprob_dict.items()
                }
                logprobs_list.append(top_logprobs)

        return {
            "tokens": [list(lp.keys())[0] if lp else None for lp in logprobs_list],
            "token_logprobs": [
                list(lp.values())[0] if lp else None for lp in logprobs_list
            ],
            "top_logprobs": logprobs_list,
        }


# Global engine instance (singleton pattern)
_engine_instance: VLLMEngine | None = None
_engine_lock = asyncio.Lock()


async def get_engine() -> VLLMEngine:
    """Get or create the global engine instance.

    Returns:
        The singleton VLLMEngine instance.
    """
    global _engine_instance

    async with _engine_lock:
        if _engine_instance is None:
            _engine_instance = VLLMEngine()
        return _engine_instance


async def initialize_engine() -> VLLMEngine:
    """Initialize and start the global engine.

    Should be called during application startup.

    Returns:
        The initialized and started VLLMEngine.
    """
    engine = await get_engine()
    await engine.start()
    return engine


async def shutdown_engine() -> None:
    """Shutdown the global engine.

    Should be called during application shutdown.
    """
    global _engine_instance

    async with _engine_lock:
        if _engine_instance is not None:
            await _engine_instance.stop()
            _engine_instance = None
