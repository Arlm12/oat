"""OpenAI SDK auto-instrumentation (including OpenAI-compatible endpoints)."""

from __future__ import annotations

import functools
from typing import Any, Optional, Tuple

from ..media import extract_openai_multimodal_artifacts
from ..models import LLMUsage, SpanStatus, SpanType
from ..pricing import build_llm_usage
from ..providers import resolve_service_provider
from ..tracer import _span_id, get_tracer

_original_chat_create = None
_original_chat_create_async = None
_original_embeddings_create = None
_original_embeddings_create_async = None
_patched = False


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        chunks = []
        for item in value:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and item.get("text"):
                    chunks.append(str(item["text"]))
                elif item.get("text"):
                    chunks.append(str(item["text"]))
            elif getattr(item, "type", None) == "text" and getattr(item, "text", None):
                chunks.append(str(item.text))
        return "\n".join([c for c in chunks if c]).strip()
    return str(value)


def _extract_output_text(response: Any) -> str:
    # Chat Completions API.
    choices = getattr(response, "choices", None)
    if choices:
        first = choices[0]
        message = getattr(first, "message", None)
        if message is not None:
            content = getattr(message, "content", None)
            text = _as_text(content)
            if text:
                return text
    # Responses API style fallback.
    text = getattr(response, "output_text", None)
    if text:
        return _as_text(text)
    if isinstance(response, dict):
        return _as_text(response.get("output_text") or response.get("content"))
    return ""


def _extract_usage(response: Any) -> Tuple[int, int, int]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return 0, 0, 0

    if isinstance(usage, dict):
        prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
        return prompt_tokens, completion_tokens, total_tokens

    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens))
    return prompt_tokens, completion_tokens, total_tokens


def _resolve_provider(args: tuple, kwargs: dict, model: str) -> str:
    base_url = None
    provider_hint = kwargs.get("provider") or kwargs.get("service_provider")

    if args:
        resource = args[0]
        client = getattr(resource, "_client", None)
        if client is not None:
            client_base = getattr(client, "base_url", None) or getattr(client, "_base_url", None)
            if client_base is not None:
                base_url = str(client_base)
    if kwargs.get("base_url"):
        base_url = str(kwargs.get("base_url"))

    return resolve_service_provider(
        model=model,
        base_url=base_url,
        provider_hint=provider_hint,
        default="openai",
    )


def _assign_usage(span: Any, model: str, provider: str, prompt_tokens: int, completion_tokens: int, total_tokens: int) -> None:
    usage = build_llm_usage(
        model=model,
        provider=provider,
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


def _record_input_artifacts(tracer: Any, span: Any, model: str, messages: Any, kwargs: dict) -> tuple[int, int]:
    inputs = {
        "model": model,
        "messages": messages,
        "temperature": kwargs.get("temperature"),
        "max_tokens": kwargs.get("max_tokens"),
    }
    tracer.record_artifact(
        span,
        role="input.message",
        content_type="application/json",
        content=inputs,
        preview={"model": model, "message_count": len(messages) if isinstance(messages, list) else 0},
    )

    image_count = 0
    audio_count = 0
    for artifact in extract_openai_multimodal_artifacts(messages):
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


def _record_output_artifacts(tracer: Any, span: Any, text: str) -> None:
    if not text:
        return
    tracer.record_artifact(
        span,
        role="output.text",
        content_type="text/plain",
        content=text,
        inline_text=text,
    )


def _finish_span_with_media(tracer: Any, span: Any, image_count: int, audio_count: int, model: str, provider: str) -> None:
    if image_count or audio_count:
        tracer.record_artifact(
            span,
            role="derived.media_analysis",
            content_type="application/json",
            preview={"image_count": image_count, "audio_count": audio_count, "model": model},
            metadata={"source": "openai.chat.completions.create", "provider": provider},
        )


def _iter_stream_sync(tracer, span, stream, model, provider, image_count, audio_count):
    """Wrap a sync OpenAI stream; finish the span when iteration is complete."""
    text_parts: List[str] = []
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    try:
        for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if choices:
                delta = getattr(choices[0], "delta", None)
                content = getattr(delta, "content", None) if delta else None
                if content:
                    text_parts.append(content)
            # OpenAI sends usage in the final chunk when stream_options.include_usage=True
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                total_tokens = int(getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens))
            yield chunk
    except Exception as exc:
        tracer.finish_span(span, SpanStatus.ERROR, error=exc)
        raise
    else:
        output_text = "".join(text_parts)
        _record_output_artifacts(tracer, span, output_text)
        _finish_span_with_media(tracer, span, image_count, audio_count, model, provider)
        _assign_usage(span, model, provider, prompt_tokens, completion_tokens, total_tokens)
        tracer.finish_span(span, SpanStatus.SUCCESS)


async def _iter_stream_async(tracer, span, stream, model, provider, image_count, audio_count):
    """Wrap an async OpenAI stream; finish the span when iteration is complete."""
    text_parts: List[str] = []
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    try:
        async for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if choices:
                delta = getattr(choices[0], "delta", None)
                content = getattr(delta, "content", None) if delta else None
                if content:
                    text_parts.append(content)
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                total_tokens = int(getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens))
            yield chunk
    except Exception as exc:
        tracer.finish_span(span, SpanStatus.ERROR, error=exc)
        raise
    else:
        output_text = "".join(text_parts)
        _record_output_artifacts(tracer, span, output_text)
        _finish_span_with_media(tracer, span, image_count, audio_count, model, provider)
        _assign_usage(span, model, provider, prompt_tokens, completion_tokens, total_tokens)
        tracer.finish_span(span, SpanStatus.SUCCESS)


def _wrap_chat_completion(original_func):
    @functools.wraps(original_func)
    def wrapper(*args, **kwargs):
        tracer = get_tracer()
        model = str(kwargs.get("model") or "unknown")
        provider = _resolve_provider(args, kwargs, model)
        messages = kwargs.get("messages", [])
        is_streaming = bool(kwargs.get("stream"))

        span = tracer.create_span(
            name="openai.chat.completions.create",
            span_type=SpanType.LLM,
            model=model,
            provider=provider,
        )
        token = _span_id.set(span.span_id)
        image_count, audio_count = _record_input_artifacts(tracer, span, model, messages, kwargs)
        # Reset parent context so sibling spans created after this call don't
        # become children of this LLM span (same behaviour as non-streaming).
        _span_id.reset(token)

        try:
            response = original_func(*args, **kwargs)
            if is_streaming:
                # Return a generator; the span is finished inside it.
                return _iter_stream_sync(tracer, span, response, model, provider, image_count, audio_count)
            prompt_tokens, completion_tokens, total_tokens = _extract_usage(response)
            _assign_usage(span, model, provider, prompt_tokens, completion_tokens, total_tokens)
            output_text = _extract_output_text(response)
            _record_output_artifacts(tracer, span, output_text)
            _finish_span_with_media(tracer, span, image_count, audio_count, model, provider)
            tracer.finish_span(span, SpanStatus.SUCCESS)
            return response
        except Exception as exc:
            tracer.finish_span(span, SpanStatus.ERROR, error=exc)
            raise

    return wrapper


def _wrap_chat_completion_async(original_func):
    @functools.wraps(original_func)
    async def wrapper(*args, **kwargs):
        tracer = get_tracer()
        model = str(kwargs.get("model") or "unknown")
        provider = _resolve_provider(args, kwargs, model)
        messages = kwargs.get("messages", [])
        is_streaming = bool(kwargs.get("stream"))

        span = tracer.create_span(
            name="openai.chat.completions.create",
            span_type=SpanType.LLM,
            model=model,
            provider=provider,
        )
        token = _span_id.set(span.span_id)
        image_count, audio_count = _record_input_artifacts(tracer, span, model, messages, kwargs)
        _span_id.reset(token)

        try:
            response = await original_func(*args, **kwargs)
            if is_streaming:
                return _iter_stream_async(tracer, span, response, model, provider, image_count, audio_count)
            prompt_tokens, completion_tokens, total_tokens = _extract_usage(response)
            _assign_usage(span, model, provider, prompt_tokens, completion_tokens, total_tokens)
            output_text = _extract_output_text(response)
            _record_output_artifacts(tracer, span, output_text)
            _finish_span_with_media(tracer, span, image_count, audio_count, model, provider)
            tracer.finish_span(span, SpanStatus.SUCCESS)
            return response
        except Exception as exc:
            tracer.finish_span(span, SpanStatus.ERROR, error=exc)
            raise

    return wrapper


def _wrap_embeddings(original_func):
    @functools.wraps(original_func)
    def wrapper(*args, **kwargs):
        tracer = get_tracer()
        model = str(kwargs.get("model") or "unknown")
        provider = _resolve_provider(args, kwargs, model)
        input_value = kwargs.get("input", "")
        input_preview = input_value[:500] if isinstance(input_value, str) else f"[{len(input_value)} items]" if isinstance(input_value, list) else str(type(input_value).__name__)

        span = tracer.create_span(
            name="openai.embeddings.create",
            span_type=SpanType.EMBEDDING,
            model=model,
            provider=provider,
        )
        token = _span_id.set(span.span_id)
        tracer.record_artifact(span, role="input.text", content_type="text/plain", inline_text=input_preview, preview={"model": model})

        try:
            response = original_func(*args, **kwargs)
            prompt_tokens, _, total_tokens = _extract_usage(response)
            _assign_usage(span, model, provider, prompt_tokens, 0, total_tokens or prompt_tokens)
            dims = len(response.data[0].embedding) if getattr(response, "data", None) else 0
            tracer.record_artifact(span, role="output.embedding", content_type="application/json", preview={"dimensions": dims})
            tracer.finish_span(span, SpanStatus.SUCCESS)
            return response
        except Exception as exc:
            tracer.finish_span(span, SpanStatus.ERROR, error=exc)
            raise
        finally:
            _span_id.reset(token)

    return wrapper


def _wrap_embeddings_async(original_func):
    @functools.wraps(original_func)
    async def wrapper(*args, **kwargs):
        tracer = get_tracer()
        model = str(kwargs.get("model") or "unknown")
        provider = _resolve_provider(args, kwargs, model)
        input_value = kwargs.get("input", "")
        input_preview = input_value[:500] if isinstance(input_value, str) else f"[{len(input_value)} items]" if isinstance(input_value, list) else str(type(input_value).__name__)

        span = tracer.create_span(
            name="openai.embeddings.create",
            span_type=SpanType.EMBEDDING,
            model=model,
            provider=provider,
        )
        token = _span_id.set(span.span_id)
        tracer.record_artifact(span, role="input.text", content_type="text/plain", inline_text=input_preview, preview={"model": model})

        try:
            response = await original_func(*args, **kwargs)
            prompt_tokens, _, total_tokens = _extract_usage(response)
            _assign_usage(span, model, provider, prompt_tokens, 0, total_tokens or prompt_tokens)
            dims = len(response.data[0].embedding) if getattr(response, "data", None) else 0
            tracer.record_artifact(span, role="output.embedding", content_type="application/json", preview={"dimensions": dims})
            tracer.finish_span(span, SpanStatus.SUCCESS)
            return response
        except Exception as exc:
            tracer.finish_span(span, SpanStatus.ERROR, error=exc)
            raise
        finally:
            _span_id.reset(token)

    return wrapper


def patch_openai():
    """Patch OpenAI SDK create methods for tracing."""
    global _original_chat_create, _original_chat_create_async
    global _original_embeddings_create, _original_embeddings_create_async
    global _patched

    if _patched:
        return

    try:
        from openai.resources import embeddings
        from openai.resources.chat import completions as chat_completions

        _original_chat_create = chat_completions.Completions.create
        chat_completions.Completions.create = _wrap_chat_completion(_original_chat_create)

        _original_chat_create_async = chat_completions.AsyncCompletions.create
        chat_completions.AsyncCompletions.create = _wrap_chat_completion_async(_original_chat_create_async)

        _original_embeddings_create = embeddings.Embeddings.create
        embeddings.Embeddings.create = _wrap_embeddings(_original_embeddings_create)

        _original_embeddings_create_async = embeddings.AsyncEmbeddings.create
        embeddings.AsyncEmbeddings.create = _wrap_embeddings_async(_original_embeddings_create_async)

        _patched = True
        print("[OAT] OpenAI patched successfully")
    except ImportError:
        print("[OAT] OpenAI not installed, skipping patch")
    except Exception as exc:
        print(f"[OAT] Failed to patch OpenAI: {exc}")


def unpatch_openai():
    """Restore original OpenAI SDK methods."""
    global _patched

    if not _patched:
        return

    try:
        from openai.resources import embeddings
        from openai.resources.chat import completions as chat_completions

        if _original_chat_create:
            chat_completions.Completions.create = _original_chat_create
        if _original_chat_create_async:
            chat_completions.AsyncCompletions.create = _original_chat_create_async
        if _original_embeddings_create:
            embeddings.Embeddings.create = _original_embeddings_create
        if _original_embeddings_create_async:
            embeddings.AsyncEmbeddings.create = _original_embeddings_create_async
        _patched = False
    except Exception:
        pass
