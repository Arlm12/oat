"""Anthropic SDK auto-instrumentation."""

from __future__ import annotations

import functools
from typing import Any

from ..media import extract_anthropic_multimodal_artifacts
from ..models import LLMUsage, SpanStatus, SpanType
from ..pricing import build_llm_usage
from ..tracer import _span_id, get_tracer

_original_messages_create = None
_original_messages_create_async = None
_patched = False


def _extract_response_text(response: Any) -> str:
    content = getattr(response, "content", None)
    if not content:
        return ""
    chunks = []
    for item in content:
        if getattr(item, "type", None) == "text" and getattr(item, "text", None):
            chunks.append(str(item.text))
        elif isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
            chunks.append(str(item["text"]))
    return "\n".join(chunks).strip()


def _extract_usage(response: Any) -> tuple[int, int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0, 0
    prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    total_tokens = prompt_tokens + completion_tokens
    return prompt_tokens, completion_tokens, total_tokens


def _assign_usage(span: Any, model: str, prompt_tokens: int, completion_tokens: int, total_tokens: int) -> None:
    usage = build_llm_usage(
        model=model,
        provider="anthropic",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    usage["total_tokens"] = int(total_tokens or usage["total_tokens"])
    span.usage = LLMUsage(
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        total_tokens=usage["total_tokens"],
        prompt_cost=usage["prompt_cost"],
        completion_cost=usage["completion_cost"],
        total_cost=usage["total_cost"],
        pricing_status=usage["pricing_status"],
    )


def _record_multimodal_inputs(tracer: Any, span: Any, messages: Any) -> tuple[int, int]:
    image_count = 0
    audio_count = 0
    for artifact in extract_anthropic_multimodal_artifacts(messages):
        tracer.record_artifact(
            span,
            role=artifact["role"],
            content_type=artifact["content_type"],
            content=artifact.get("content"),
            preview=artifact.get("preview"),
            metadata=artifact.get("metadata"),
            inline_text=artifact.get("inline_text"),
        )
        if artifact["role"] == "input.image":
            image_count += 1
        if artifact["role"] == "input.audio":
            audio_count += 1
    return image_count, audio_count


def _iter_anthropic_stream_sync(tracer, span, stream, model, image_count, audio_count):
    """Consume a sync Anthropic Stream[MessageStreamEvent] and finish the span."""
    text_parts: list = []
    prompt_tokens = 0
    completion_tokens = 0
    try:
        for event in stream:
            event_type = getattr(event, "type", None)
            # Accumulate text deltas.
            if event_type == "content_block_delta":
                delta = getattr(event, "delta", None)
                if getattr(delta, "type", None) == "text_delta":
                    text_parts.append(getattr(delta, "text", "") or "")
            # Input token count arrives in message_start.
            elif event_type == "message_start":
                msg = getattr(event, "message", None)
                usage = getattr(msg, "usage", None) if msg else None
                if usage:
                    prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0)
            # Output token count arrives in message_delta.
            elif event_type == "message_delta":
                usage = getattr(event, "usage", None)
                if usage:
                    completion_tokens = int(getattr(usage, "output_tokens", 0) or 0)
            yield event
    except Exception as exc:
        tracer.finish_span(span, SpanStatus.ERROR, error=exc)
        raise
    else:
        output_text = "".join(text_parts)
        if output_text:
            tracer.record_artifact(span, role="output.text", content_type="text/plain", content=output_text, inline_text=output_text)
        if image_count or audio_count:
            tracer.record_artifact(span, role="derived.media_analysis", content_type="application/json",
                                   preview={"image_count": image_count, "audio_count": audio_count, "model": model},
                                   metadata={"source": "anthropic.messages.create"})
        _assign_usage(span, model, prompt_tokens, completion_tokens, prompt_tokens + completion_tokens)
        tracer.finish_span(span, SpanStatus.SUCCESS)


async def _iter_anthropic_stream_async(tracer, span, stream, model, image_count, audio_count):
    """Consume an async Anthropic stream and finish the span."""
    text_parts: list = []
    prompt_tokens = 0
    completion_tokens = 0
    try:
        async for event in stream:
            event_type = getattr(event, "type", None)
            if event_type == "content_block_delta":
                delta = getattr(event, "delta", None)
                if getattr(delta, "type", None) == "text_delta":
                    text_parts.append(getattr(delta, "text", "") or "")
            elif event_type == "message_start":
                msg = getattr(event, "message", None)
                usage = getattr(msg, "usage", None) if msg else None
                if usage:
                    prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0)
            elif event_type == "message_delta":
                usage = getattr(event, "usage", None)
                if usage:
                    completion_tokens = int(getattr(usage, "output_tokens", 0) or 0)
            yield event
    except Exception as exc:
        tracer.finish_span(span, SpanStatus.ERROR, error=exc)
        raise
    else:
        output_text = "".join(text_parts)
        if output_text:
            tracer.record_artifact(span, role="output.text", content_type="text/plain", content=output_text, inline_text=output_text)
        if image_count or audio_count:
            tracer.record_artifact(span, role="derived.media_analysis", content_type="application/json",
                                   preview={"image_count": image_count, "audio_count": audio_count, "model": model},
                                   metadata={"source": "anthropic.messages.create"})
        _assign_usage(span, model, prompt_tokens, completion_tokens, prompt_tokens + completion_tokens)
        tracer.finish_span(span, SpanStatus.SUCCESS)


def _wrap_messages_create(original_func):
    @functools.wraps(original_func)
    def wrapper(*args, **kwargs):
        tracer = get_tracer()
        model = str(kwargs.get("model") or "unknown")
        messages = kwargs.get("messages", [])
        system_prompt = kwargs.get("system")
        is_streaming = bool(kwargs.get("stream"))

        span = tracer.create_span(
            name="anthropic.messages.create",
            span_type=SpanType.LLM,
            model=model,
            provider="anthropic",
        )
        token = _span_id.set(span.span_id)

        input_payload = {"model": model, "system": system_prompt, "messages": messages}
        tracer.record_artifact(
            span,
            role="input.message",
            content_type="application/json",
            content=input_payload,
            preview={"model": model, "message_count": len(messages) if isinstance(messages, list) else 0},
        )
        image_count, audio_count = _record_multimodal_inputs(tracer, span, messages)
        _span_id.reset(token)

        try:
            response = original_func(*args, **kwargs)
            if is_streaming:
                return _iter_anthropic_stream_sync(tracer, span, response, model, image_count, audio_count)
            prompt_tokens, completion_tokens, total_tokens = _extract_usage(response)
            _assign_usage(span, model, prompt_tokens, completion_tokens, total_tokens)
            text = _extract_response_text(response)
            if text:
                tracer.record_artifact(span, role="output.text", content_type="text/plain", content=text, inline_text=text)
            if image_count or audio_count:
                tracer.record_artifact(
                    span,
                    role="derived.media_analysis",
                    content_type="application/json",
                    preview={"image_count": image_count, "audio_count": audio_count, "model": model},
                    metadata={"source": "anthropic.messages.create"},
                )
            tracer.finish_span(span, SpanStatus.SUCCESS)
            return response
        except Exception as exc:
            tracer.finish_span(span, SpanStatus.ERROR, error=exc)
            raise

    return wrapper


def _wrap_messages_create_async(original_func):
    @functools.wraps(original_func)
    async def wrapper(*args, **kwargs):
        tracer = get_tracer()
        model = str(kwargs.get("model") or "unknown")
        messages = kwargs.get("messages", [])
        system_prompt = kwargs.get("system")
        is_streaming = bool(kwargs.get("stream"))

        span = tracer.create_span(
            name="anthropic.messages.create",
            span_type=SpanType.LLM,
            model=model,
            provider="anthropic",
        )
        token = _span_id.set(span.span_id)

        input_payload = {"model": model, "system": system_prompt, "messages": messages}
        tracer.record_artifact(
            span,
            role="input.message",
            content_type="application/json",
            content=input_payload,
            preview={"model": model, "message_count": len(messages) if isinstance(messages, list) else 0},
        )
        image_count, audio_count = _record_multimodal_inputs(tracer, span, messages)
        _span_id.reset(token)

        try:
            response = await original_func(*args, **kwargs)
            if is_streaming:
                return _iter_anthropic_stream_async(tracer, span, response, model, image_count, audio_count)
            prompt_tokens, completion_tokens, total_tokens = _extract_usage(response)
            _assign_usage(span, model, prompt_tokens, completion_tokens, total_tokens)
            text = _extract_response_text(response)
            if text:
                tracer.record_artifact(span, role="output.text", content_type="text/plain", content=text, inline_text=text)
            if image_count or audio_count:
                tracer.record_artifact(
                    span,
                    role="derived.media_analysis",
                    content_type="application/json",
                    preview={"image_count": image_count, "audio_count": audio_count, "model": model},
                    metadata={"source": "anthropic.messages.create"},
                )
            tracer.finish_span(span, SpanStatus.SUCCESS)
            return response
        except Exception as exc:
            tracer.finish_span(span, SpanStatus.ERROR, error=exc)
            raise

    return wrapper


def patch_anthropic():
    """Patch Anthropic message creation methods."""
    global _original_messages_create, _original_messages_create_async, _patched

    if _patched:
        return

    try:
        from anthropic.resources import messages as messages_resource

        _original_messages_create = messages_resource.Messages.create
        messages_resource.Messages.create = _wrap_messages_create(_original_messages_create)

        _original_messages_create_async = messages_resource.AsyncMessages.create
        messages_resource.AsyncMessages.create = _wrap_messages_create_async(_original_messages_create_async)
        _patched = True
        print("[OAT] Anthropic patched successfully")
    except ImportError:
        print("[OAT] Anthropic not installed, skipping patch")
    except Exception as exc:
        print(f"[OAT] Failed to patch Anthropic: {exc}")


def unpatch_anthropic():
    global _patched

    if not _patched:
        return

    try:
        from anthropic.resources import messages as messages_resource

        if _original_messages_create:
            messages_resource.Messages.create = _original_messages_create
        if _original_messages_create_async:
            messages_resource.AsyncMessages.create = _original_messages_create_async
        _patched = False
    except Exception:
        pass
