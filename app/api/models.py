"""
Pydantic models for OpenAI-compatible API.

These models match the OpenAI API specification for /v1/completions
and /v1/chat/completions endpoints.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class FinishReason(str, Enum):
    """Reasons why the model stopped generating tokens."""

    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"


# =============================================================================
# Completion Request/Response Models (OpenAI /v1/completions compatible)
# =============================================================================


class CompletionRequest(BaseModel):
    """Request model for /v1/completions endpoint.

    Follows OpenAI API specification for text completions.
    """

    model: str = Field(
        ...,
        description="ID of the model to use",
        examples=["TinyLlama/TinyLlama-1.1B-Chat-v1.0"],
    )
    prompt: str | list[str] = Field(
        ...,
        description="The prompt(s) to generate completions for",
        examples=["Hello, how are you?"],
    )
    max_tokens: int = Field(
        default=256,
        ge=1,
        le=4096,
        description="Maximum number of tokens to generate",
    )
    temperature: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature between 0 and 2",
    )
    top_p: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling probability",
    )
    n: int = Field(
        default=1,
        ge=1,
        le=8,
        description="Number of completions to generate",
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream partial results",
    )
    stop: str | list[str] | None = Field(
        default=None,
        description="Sequences where the API will stop generating",
    )
    presence_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Penalty for new tokens based on presence in text so far",
    )
    frequency_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Penalty for new tokens based on frequency in text so far",
    )
    user: str | None = Field(
        default=None,
        description="Unique identifier for the end-user",
    )
    logprobs: int | None = Field(
        default=None,
        ge=0,
        le=5,
        description="Include log probabilities on the most likely tokens",
    )
    echo: bool = Field(
        default=False,
        description="Echo back the prompt in addition to the completion",
    )
    best_of: int = Field(
        default=1,
        ge=1,
        le=8,
        description="Generate best_of completions and return the best",
    )

    @field_validator("stop", mode="before")
    @classmethod
    def validate_stop(cls, v: str | list[str] | None) -> list[str] | None:
        """Convert stop to list if string provided."""
        if isinstance(v, str):
            return [v]
        return v


class CompletionChoice(BaseModel):
    """A single completion choice."""

    text: str = Field(..., description="The generated text")
    index: int = Field(..., description="Index of this choice")
    logprobs: dict[str, Any] | None = Field(
        default=None, description="Log probabilities"
    )
    finish_reason: FinishReason | None = Field(
        default=None, description="Reason generation stopped"
    )


class UsageInfo(BaseModel):
    """Token usage information."""

    prompt_tokens: int = Field(..., description="Number of tokens in the prompt")
    completion_tokens: int = Field(..., description="Number of tokens in the completion")
    total_tokens: int = Field(..., description="Total tokens used")


class CompletionResponse(BaseModel):
    """Response model for /v1/completions endpoint."""

    id: str = Field(
        default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:24]}",
        description="Unique identifier for the completion",
    )
    object: Literal["text_completion"] = Field(
        default="text_completion",
        description="Object type",
    )
    created: int = Field(
        default_factory=lambda: int(time.time()),
        description="Unix timestamp of creation",
    )
    model: str = Field(..., description="Model used for completion")
    choices: list[CompletionChoice] = Field(..., description="Completion choices")
    usage: UsageInfo = Field(..., description="Token usage information")


class CompletionStreamChoice(BaseModel):
    """A single streaming completion choice."""

    text: str = Field(..., description="The generated text delta")
    index: int = Field(..., description="Index of this choice")
    logprobs: dict[str, Any] | None = Field(default=None, description="Log probabilities")
    finish_reason: FinishReason | None = Field(
        default=None, description="Reason generation stopped"
    )


class CompletionStreamResponse(BaseModel):
    """Streaming response model for /v1/completions endpoint."""

    id: str = Field(
        default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:24]}",
        description="Unique identifier for the completion",
    )
    object: Literal["text_completion"] = Field(
        default="text_completion",
        description="Object type",
    )
    created: int = Field(
        default_factory=lambda: int(time.time()),
        description="Unix timestamp of creation",
    )
    model: str = Field(..., description="Model used for completion")
    choices: list[CompletionStreamChoice] = Field(..., description="Completion choices")


# =============================================================================
# Chat Completion Request/Response Models (OpenAI /v1/chat/completions compatible)
# =============================================================================


class ChatMessageRole(str, Enum):
    """Valid roles for chat messages."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """A single chat message."""

    role: ChatMessageRole = Field(..., description="Role of the message author")
    content: str = Field(..., description="Content of the message")
    name: str | None = Field(default=None, description="Optional name of the author")


class ChatCompletionRequest(BaseModel):
    """Request model for /v1/chat/completions endpoint.

    Follows OpenAI API specification for chat completions.
    """

    model: str = Field(
        ...,
        description="ID of the model to use",
        examples=["TinyLlama/TinyLlama-1.1B-Chat-v1.0"],
    )
    messages: list[ChatMessage] = Field(
        ...,
        min_length=1,
        description="List of messages in the conversation",
    )
    max_tokens: int = Field(
        default=256,
        ge=1,
        le=4096,
        description="Maximum number of tokens to generate",
    )
    temperature: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature between 0 and 2",
    )
    top_p: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling probability",
    )
    n: int = Field(
        default=1,
        ge=1,
        le=8,
        description="Number of chat completions to generate",
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream partial message deltas",
    )
    stop: str | list[str] | None = Field(
        default=None,
        description="Sequences where the API will stop generating",
    )
    presence_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Penalty for new tokens based on presence in text so far",
    )
    frequency_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Penalty for new tokens based on frequency in text so far",
    )
    user: str | None = Field(
        default=None,
        description="Unique identifier for the end-user",
    )

    @field_validator("stop", mode="before")
    @classmethod
    def validate_stop(cls, v: str | list[str] | None) -> list[str] | None:
        """Convert stop to list if string provided."""
        if isinstance(v, str):
            return [v]
        return v


class ChatCompletionMessage(BaseModel):
    """A message in a chat completion response."""

    role: ChatMessageRole = Field(..., description="Role of the message author")
    content: str = Field(..., description="Content of the message")


class ChatCompletionChoice(BaseModel):
    """A single chat completion choice."""

    index: int = Field(..., description="Index of this choice")
    message: ChatCompletionMessage = Field(..., description="The generated message")
    finish_reason: FinishReason | None = Field(
        default=None, description="Reason generation stopped"
    )


class ChatCompletionResponse(BaseModel):
    """Response model for /v1/chat/completions endpoint."""

    id: str = Field(
        default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}",
        description="Unique identifier for the chat completion",
    )
    object: Literal["chat.completion"] = Field(
        default="chat.completion",
        description="Object type",
    )
    created: int = Field(
        default_factory=lambda: int(time.time()),
        description="Unix timestamp of creation",
    )
    model: str = Field(..., description="Model used for completion")
    choices: list[ChatCompletionChoice] = Field(..., description="Completion choices")
    usage: UsageInfo = Field(..., description="Token usage information")


class ChatCompletionDelta(BaseModel):
    """A delta update for streaming chat completion."""

    role: ChatMessageRole | None = Field(default=None, description="Role of the message author")
    content: str | None = Field(default=None, description="Content delta")


class ChatCompletionStreamChoice(BaseModel):
    """A single streaming chat completion choice."""

    index: int = Field(..., description="Index of this choice")
    delta: ChatCompletionDelta = Field(..., description="The message delta")
    finish_reason: FinishReason | None = Field(
        default=None, description="Reason generation stopped"
    )


class ChatCompletionStreamResponse(BaseModel):
    """Streaming response model for /v1/chat/completions endpoint."""

    id: str = Field(
        default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}",
        description="Unique identifier for the chat completion",
    )
    object: Literal["chat.completion.chunk"] = Field(
        default="chat.completion.chunk",
        description="Object type",
    )
    created: int = Field(
        default_factory=lambda: int(time.time()),
        description="Unix timestamp of creation",
    )
    model: str = Field(..., description="Model used for completion")
    choices: list[ChatCompletionStreamChoice] = Field(..., description="Completion choices")


# =============================================================================
# Model Information Models
# =============================================================================


class ModelInfo(BaseModel):
    """Information about an available model."""

    id: str = Field(..., description="Model identifier")
    object: Literal["model"] = Field(default="model", description="Object type")
    created: int = Field(
        default_factory=lambda: int(time.time()),
        description="Unix timestamp of creation",
    )
    owned_by: str = Field(default="local", description="Organization that owns the model")


class ModelList(BaseModel):
    """List of available models."""

    object: Literal["list"] = Field(default="list", description="Object type")
    data: list[ModelInfo] = Field(..., description="List of models")


# =============================================================================
# Error Models
# =============================================================================


class ErrorDetail(BaseModel):
    """Details of an API error."""

    message: str = Field(..., description="Error message")
    type: str = Field(..., description="Error type")
    param: str | None = Field(default=None, description="Parameter that caused the error")
    code: str | None = Field(default=None, description="Error code")


class ErrorResponse(BaseModel):
    """Error response matching OpenAI format."""

    error: ErrorDetail = Field(..., description="Error details")
