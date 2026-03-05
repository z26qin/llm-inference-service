"""
API Routes for OpenAI-compatible endpoints.

Implements /v1/completions and /v1/chat/completions with both
streaming (SSE) and non-streaming responses.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from app.api.models import (
    ChatCompletionChoice,
    ChatCompletionDelta,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessageRole,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    CompletionStreamChoice,
    CompletionStreamResponse,
    ErrorDetail,
    ErrorResponse,
    FinishReason,
    ModelInfo,
    ModelList,
    UsageInfo,
)
from app.engine.vllm_engine import GenerationRequest, get_engine

router = APIRouter(prefix="/v1", tags=["OpenAI Compatible API"])


def _build_chat_prompt(request: ChatCompletionRequest) -> str:
    """Build a prompt string from chat messages.

    Uses a simple template format compatible with most chat models.
    For TinyLlama specifically, uses the ChatML format.

    Args:
        request: The chat completion request.

    Returns:
        Formatted prompt string.
    """
    # ChatML format (used by TinyLlama and many other models)
    prompt_parts: list[str] = []

    for message in request.messages:
        role = message.role.value
        content = message.content

        if role == "system":
            prompt_parts.append(f"<|system|>\n{content}</s>")
        elif role == "user":
            prompt_parts.append(f"<|user|>\n{content}</s>")
        elif role == "assistant":
            prompt_parts.append(f"<|assistant|>\n{content}</s>")

    # Add assistant prefix for the response
    prompt_parts.append("<|assistant|>\n")

    return "\n".join(prompt_parts)


def _map_finish_reason(reason: str | None) -> FinishReason | None:
    """Map vLLM finish reason to OpenAI format.

    Args:
        reason: vLLM finish reason string.

    Returns:
        OpenAI FinishReason enum value.
    """
    if reason is None:
        return None

    mapping = {
        "stop": FinishReason.STOP,
        "length": FinishReason.LENGTH,
        "abort": FinishReason.STOP,
    }

    return mapping.get(reason.lower(), FinishReason.STOP)


# =============================================================================
# Completions Endpoints
# =============================================================================


@router.post(
    "/completions",
    response_model=CompletionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
    summary="Create a completion",
    description="Creates a completion for the provided prompt and parameters.",
)
async def create_completion(
    request: CompletionRequest,
    http_request: Request,
) -> CompletionResponse | EventSourceResponse:
    """Handle completion requests with optional streaming.

    Args:
        request: The completion request.
        http_request: The HTTP request object.

    Returns:
        CompletionResponse for non-streaming, EventSourceResponse for streaming.

    Raises:
        HTTPException: On validation or processing errors.
    """
    engine = await get_engine()

    if not engine.is_ready():
        raise HTTPException(
            status_code=503,
            detail=ErrorDetail(
                message="Model not ready",
                type="service_unavailable",
                code="model_not_ready",
            ).model_dump(),
        )

    # Handle list of prompts (just take first for simplicity)
    prompt = request.prompt if isinstance(request.prompt, str) else request.prompt[0]

    request_id = f"cmpl-{uuid.uuid4().hex[:24]}"

    gen_request = GenerationRequest(
        request_id=request_id,
        prompt=prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=request.stop,  # type: ignore
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
        n=request.n,
        logprobs=request.logprobs,
        echo=request.echo,
        best_of=request.best_of,
    )

    if request.stream:
        return EventSourceResponse(
            _stream_completion(gen_request, engine.model_name, request_id),
            media_type="text/event-stream",
        )

    try:
        output = await engine.generate(gen_request)

        return CompletionResponse(
            id=request_id,
            created=int(time.time()),
            model=engine.model_name,
            choices=[
                CompletionChoice(
                    text=output.text,
                    index=0,
                    logprobs=output.logprobs,
                    finish_reason=_map_finish_reason(output.finish_reason),
                )
            ],
            usage=UsageInfo(
                prompt_tokens=output.prompt_tokens,
                completion_tokens=output.completion_tokens,
                total_tokens=output.prompt_tokens + output.completion_tokens,
            ),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ErrorDetail(
                message=str(e),
                type="internal_error",
                code="generation_failed",
            ).model_dump(),
        ) from e


async def _stream_completion(
    request: GenerationRequest,
    model_name: str,
    completion_id: str,
) -> AsyncIterator[dict]:
    """Stream completion tokens as SSE events.

    Args:
        request: The generation request.
        model_name: Name of the model.
        completion_id: Unique completion ID.

    Yields:
        SSE event dictionaries.
    """
    engine = await get_engine()
    created = int(time.time())

    try:
        async for output in engine.generate_stream(request):
            response = CompletionStreamResponse(
                id=completion_id,
                created=created,
                model=model_name,
                choices=[
                    CompletionStreamChoice(
                        text=output.text,
                        index=0,
                        logprobs=output.logprobs,
                        finish_reason=_map_finish_reason(output.finish_reason),
                    )
                ],
            )

            yield {"data": response.model_dump_json()}

        # Send final [DONE] message
        yield {"data": "[DONE]"}

    except Exception as e:
        error_response = ErrorResponse(
            error=ErrorDetail(
                message=str(e),
                type="internal_error",
                code="stream_error",
            )
        )
        yield {"data": error_response.model_dump_json()}


# =============================================================================
# Chat Completions Endpoints
# =============================================================================


@router.post(
    "/chat/completions",
    response_model=ChatCompletionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
    summary="Create a chat completion",
    description="Creates a completion for the provided chat conversation.",
)
async def create_chat_completion(
    request: ChatCompletionRequest,
    http_request: Request,
) -> ChatCompletionResponse | EventSourceResponse:
    """Handle chat completion requests with optional streaming.

    Args:
        request: The chat completion request.
        http_request: The HTTP request object.

    Returns:
        ChatCompletionResponse for non-streaming, EventSourceResponse for streaming.

    Raises:
        HTTPException: On validation or processing errors.
    """
    engine = await get_engine()

    if not engine.is_ready():
        raise HTTPException(
            status_code=503,
            detail=ErrorDetail(
                message="Model not ready",
                type="service_unavailable",
                code="model_not_ready",
            ).model_dump(),
        )

    # Build prompt from messages
    prompt = _build_chat_prompt(request)
    request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    gen_request = GenerationRequest(
        request_id=request_id,
        prompt=prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=request.stop,  # type: ignore
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
        n=request.n,
    )

    if request.stream:
        return EventSourceResponse(
            _stream_chat_completion(gen_request, engine.model_name, request_id),
            media_type="text/event-stream",
        )

    try:
        output = await engine.generate(gen_request)

        return ChatCompletionResponse(
            id=request_id,
            created=int(time.time()),
            model=engine.model_name,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(
                        role=ChatMessageRole.ASSISTANT,
                        content=output.text.strip(),
                    ),
                    finish_reason=_map_finish_reason(output.finish_reason),
                )
            ],
            usage=UsageInfo(
                prompt_tokens=output.prompt_tokens,
                completion_tokens=output.completion_tokens,
                total_tokens=output.prompt_tokens + output.completion_tokens,
            ),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ErrorDetail(
                message=str(e),
                type="internal_error",
                code="generation_failed",
            ).model_dump(),
        ) from e


async def _stream_chat_completion(
    request: GenerationRequest,
    model_name: str,
    completion_id: str,
) -> AsyncIterator[dict]:
    """Stream chat completion tokens as SSE events.

    Args:
        request: The generation request.
        model_name: Name of the model.
        completion_id: Unique completion ID.

    Yields:
        SSE event dictionaries.
    """
    engine = await get_engine()
    created = int(time.time())
    first_chunk = True

    try:
        async for output in engine.generate_stream(request):
            # First chunk includes role
            if first_chunk:
                delta = ChatCompletionDelta(
                    role=ChatMessageRole.ASSISTANT,
                    content=output.text,
                )
                first_chunk = False
            else:
                delta = ChatCompletionDelta(content=output.text)

            response = ChatCompletionStreamResponse(
                id=completion_id,
                created=created,
                model=model_name,
                choices=[
                    ChatCompletionStreamChoice(
                        index=0,
                        delta=delta,
                        finish_reason=_map_finish_reason(output.finish_reason),
                    )
                ],
            )

            yield {"data": response.model_dump_json()}

        # Send final [DONE] message
        yield {"data": "[DONE]"}

    except Exception as e:
        error_response = ErrorResponse(
            error=ErrorDetail(
                message=str(e),
                type="internal_error",
                code="stream_error",
            )
        )
        yield {"data": error_response.model_dump_json()}


# =============================================================================
# Models Endpoint
# =============================================================================


@router.get(
    "/models",
    response_model=ModelList,
    summary="List models",
    description="Lists the currently available models.",
)
async def list_models() -> ModelList:
    """List available models.

    Returns:
        ModelList with available model information.
    """
    engine = await get_engine()

    return ModelList(
        data=[
            ModelInfo(
                id=engine.model_name,
                created=int(time.time()),
                owned_by="local",
            )
        ]
    )


@router.get(
    "/models/{model_id}",
    response_model=ModelInfo,
    responses={404: {"model": ErrorResponse, "description": "Model not found"}},
    summary="Retrieve model",
    description="Retrieves a model instance.",
)
async def get_model(model_id: str) -> ModelInfo:
    """Get information about a specific model.

    Args:
        model_id: The model identifier.

    Returns:
        ModelInfo for the requested model.

    Raises:
        HTTPException: If model not found.
    """
    engine = await get_engine()

    if model_id != engine.model_name:
        raise HTTPException(
            status_code=404,
            detail=ErrorDetail(
                message=f"Model '{model_id}' not found",
                type="invalid_request_error",
                code="model_not_found",
            ).model_dump(),
        )

    return ModelInfo(
        id=engine.model_name,
        created=int(time.time()),
        owned_by="local",
    )
