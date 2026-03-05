"""Ollama SDK auto-instrumentation."""

from __future__ import annotations

import functools
from typing import Any

from ..models import LLMUsage, SpanStatus, SpanType
from ..pricing import build_llm_usage
from ..tracer import _span_id, get_tracer

_original_chat = None
_original_client_chat = None
_original_async_client_chat = None
_patched = False


def _extract_messages(args: tuple, kwargs: dict) -> Any:
    if "messages" in kwargs:
        return kwargs.get("messages")
    if len(args) >= 2:
        return args[1]
    return []


def _extract_model(args: tuple, kwargs: dict) -> str:
    if "model" in kwargs:
        return str(kwargs.get("model"))
    if len(args) >= 1 and isinstance(args[0], str):
        return str(args[0])
    if len(args) >= 2 and hasattr(args[0], "__class__"):
        # bound method form: self, model, ...
        maybe_model = args[1]
        if isinstance(maybe_model, str):
            return maybe_model
    return "ollama-unknown"


def _extract_output_text(response: Any) -> str:
    if response is None:
        return ""
    if isinstance(response, dict):
        message = response.get("message") or {}
        content = message.get("content")
        return str(content or "")
    message = getattr(response, "message", None)
    if message is not None:
        content = getattr(message, "content", None)
        if content:
            return str(content)
    return str(getattr(response, "response", "") or "")


def _extract_usage(response: Any) -> tuple[int, int, int]:
    if isinstance(response, dict):
        prompt_tokens = int(response.get("prompt_eval_count") or 0)
        completion_tokens = int(response.get("eval_count") or 0)
        return prompt_tokens, completion_tokens, prompt_tokens + completion_tokens
    prompt_tokens = int(getattr(response, "prompt_eval_count", 0) or 0)
    completion_tokens = int(getattr(response, "eval_count", 0) or 0)
    return prompt_tokens, completion_tokens, prompt_tokens + completion_tokens


def _record_message_artifacts(tracer: Any, span: Any, messages: Any) -> tuple[int, int]:
    image_count = 0
    audio_count = 0
    if not isinstance(messages, list):
        return image_count, audio_count
    for mi, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            tracer.record_artifact(
                span,
                role="input.text",
                content_type="text/plain",
                content=content,
                inline_text=content,
                preview={"message_index": mi},
                metadata={"message_index": mi},
            )
        for ii, image in enumerate(message.get("images") or []):
            tracer.record_artifact(
                span,
                role="input.image",
                content_type="image/*",
                content=image,
                preview={"message_index": mi, "image_index": ii},
                metadata={"message_index": mi, "image_index": ii},
            )
            image_count += 1
    return image_count, audio_count


def _is_done(chunk: Any) -> bool:
    """Return True if this is the final chunk of an Ollama streaming response."""
    if isinstance(chunk, dict):
        return bool(chunk.get("done"))
    return bool(getattr(chunk, "done", False))


def _chunk_text(chunk: Any) -> str:
    """Extract partial text from a streaming Ollama chunk."""
    if isinstance(chunk, dict):
        msg = chunk.get("message") or {}
        return str(msg.get("content") or "")
    msg = getattr(chunk, "message", None)
    if msg is not None:
        return str(getattr(msg, "content", "") or "")
    return ""


def _assign_usage(span: Any, model: str, prompt_tokens: int, completion_tokens: int, total_tokens: int) -> None:
    usage = build_llm_usage(
        model=model,
        provider="ollama",
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


def _iter_ollama_stream_sync(tracer, span, stream, model, image_count, audio_count):
    """Consume a sync Ollama streaming response and finish the span."""
    text_parts: list = []
    prompt_tokens = 0
    completion_tokens = 0
    try:
        for chunk in stream:
            text_parts.append(_chunk_text(chunk))
            if _is_done(chunk):
                prompt_tokens, completion_tokens, _ = _extract_usage(chunk)
            yield chunk
    except Exception as exc:
        tracer.finish_span(span, SpanStatus.ERROR, error=exc)
        raise
    else:
        output_text = "".join(text_parts)
        if output_text:
            tracer.record_artifact(
                span, role="output.text", content_type="text/plain",
                content=output_text, inline_text=output_text,
            )
        if image_count or audio_count:
            tracer.record_artifact(
                span, role="derived.media_analysis", content_type="application/json",
                preview={"image_count": image_count, "audio_count": audio_count, "model": model},
                metadata={"source": "ollama.chat"},
            )
        _assign_usage(span, model, prompt_tokens, completion_tokens, prompt_tokens + completion_tokens)
        tracer.finish_span(span, SpanStatus.SUCCESS)


async def _iter_ollama_stream_async(tracer, span, stream, model, image_count, audio_count):
    """Consume an async Ollama streaming response and finish the span."""
    text_parts: list = []
    prompt_tokens = 0
    completion_tokens = 0
    try:
        async for chunk in stream:
            text_parts.append(_chunk_text(chunk))
            if _is_done(chunk):
                prompt_tokens, completion_tokens, _ = _extract_usage(chunk)
            yield chunk
    except Exception as exc:
        tracer.finish_span(span, SpanStatus.ERROR, error=exc)
        raise
    else:
        output_text = "".join(text_parts)
        if output_text:
            tracer.record_artifact(
                span, role="output.text", content_type="text/plain",
                content=output_text, inline_text=output_text,
            )
        if image_count or audio_count:
            tracer.record_artifact(
                span, role="derived.media_analysis", content_type="application/json",
                preview={"image_count": image_count, "audio_count": audio_count, "model": model},
                metadata={"source": "ollama.chat"},
            )
        _assign_usage(span, model, prompt_tokens, completion_tokens, prompt_tokens + completion_tokens)
        tracer.finish_span(span, SpanStatus.SUCCESS)


def _wrap_chat(original_func):
    @functools.wraps(original_func)
    def wrapper(*args, **kwargs):
        tracer = get_tracer()
        model = _extract_model(args, kwargs)
        messages = _extract_messages(args, kwargs)
        is_streaming = bool(kwargs.get("stream"))

        span = tracer.create_span(
            name="ollama.chat",
            span_type=SpanType.LLM,
            model=model,
            provider="ollama",
        )
        token = _span_id.set(span.span_id)
        input_payload = {"model": model, "messages": messages}
        tracer.record_artifact(
            span,
            role="input.message",
            content_type="application/json",
            content=input_payload,
            preview={"model": model, "message_count": len(messages) if isinstance(messages, list) else 0},
        )
        image_count, audio_count = _record_message_artifacts(tracer, span, messages)
        _span_id.reset(token)

        try:
            response = original_func(*args, **kwargs)
            if is_streaming:
                return _iter_ollama_stream_sync(tracer, span, response, model, image_count, audio_count)
            prompt_tokens, completion_tokens, total_tokens = _extract_usage(response)
            _assign_usage(span, model, prompt_tokens, completion_tokens, total_tokens)
            output_text = _extract_output_text(response)
            if output_text:
                tracer.record_artifact(
                    span,
                    role="output.text",
                    content_type="text/plain",
                    content=output_text,
                    inline_text=output_text,
                )
            if image_count or audio_count:
                tracer.record_artifact(
                    span,
                    role="derived.media_analysis",
                    content_type="application/json",
                    preview={"image_count": image_count, "audio_count": audio_count, "model": model},
                    metadata={"source": "ollama.chat"},
                )
            tracer.finish_span(span, SpanStatus.SUCCESS)
            return response
        except Exception as exc:
            tracer.finish_span(span, SpanStatus.ERROR, error=exc)
            raise

    return wrapper


def _wrap_chat_async(original_func):
    @functools.wraps(original_func)
    async def wrapper(*args, **kwargs):
        tracer = get_tracer()
        model = _extract_model(args, kwargs)
        messages = _extract_messages(args, kwargs)
        is_streaming = bool(kwargs.get("stream"))

        span = tracer.create_span(
            name="ollama.chat",
            span_type=SpanType.LLM,
            model=model,
            provider="ollama",
        )
        token = _span_id.set(span.span_id)
        input_payload = {"model": model, "messages": messages}
        tracer.record_artifact(
            span,
            role="input.message",
            content_type="application/json",
            content=input_payload,
            preview={"model": model, "message_count": len(messages) if isinstance(messages, list) else 0},
        )
        image_count, audio_count = _record_message_artifacts(tracer, span, messages)
        _span_id.reset(token)

        try:
            response = await original_func(*args, **kwargs)
            if is_streaming:
                return _iter_ollama_stream_async(tracer, span, response, model, image_count, audio_count)
            prompt_tokens, completion_tokens, total_tokens = _extract_usage(response)
            _assign_usage(span, model, prompt_tokens, completion_tokens, total_tokens)
            output_text = _extract_output_text(response)
            if output_text:
                tracer.record_artifact(
                    span,
                    role="output.text",
                    content_type="text/plain",
                    content=output_text,
                    inline_text=output_text,
                )
            if image_count or audio_count:
                tracer.record_artifact(
                    span,
                    role="derived.media_analysis",
                    content_type="application/json",
                    preview={"image_count": image_count, "audio_count": audio_count, "model": model},
                    metadata={"source": "ollama.chat"},
                )
            tracer.finish_span(span, SpanStatus.SUCCESS)
            return response
        except Exception as exc:
            tracer.finish_span(span, SpanStatus.ERROR, error=exc)
            raise

    return wrapper


def patch_ollama():
    """Patch Ollama sync/async chat APIs."""
    global _original_chat, _original_client_chat, _original_async_client_chat, _patched

    if _patched:
        return

    try:
        import ollama

        if hasattr(ollama, "chat"):
            _original_chat = ollama.chat
            ollama.chat = _wrap_chat(_original_chat)
        if hasattr(ollama, "Client") and hasattr(ollama.Client, "chat"):
            _original_client_chat = ollama.Client.chat
            ollama.Client.chat = _wrap_chat(_original_client_chat)
        if hasattr(ollama, "AsyncClient") and hasattr(ollama.AsyncClient, "chat"):
            _original_async_client_chat = ollama.AsyncClient.chat
            ollama.AsyncClient.chat = _wrap_chat_async(_original_async_client_chat)

        _patched = True
        print("[OAT] Ollama patched successfully")
    except ImportError:
        print("[OAT] Ollama not installed, skipping patch")
    except Exception as exc:
        print(f"[OAT] Failed to patch Ollama: {exc}")


def unpatch_ollama():
    global _patched

    if not _patched:
        return

    try:
        import ollama

        if _original_chat and hasattr(ollama, "chat"):
            ollama.chat = _original_chat
        if _original_client_chat and hasattr(ollama, "Client") and hasattr(ollama.Client, "chat"):
            ollama.Client.chat = _original_client_chat
        if _original_async_client_chat and hasattr(ollama, "AsyncClient") and hasattr(ollama.AsyncClient, "chat"):
            ollama.AsyncClient.chat = _original_async_client_chat
        _patched = False
    except Exception:
        pass
