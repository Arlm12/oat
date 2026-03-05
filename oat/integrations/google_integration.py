"""Google Gemini (google-generativeai) auto-instrumentation."""

from __future__ import annotations

import functools
from typing import Any

from ..media import extract_gemini_multimodal_artifacts
from ..models import LLMUsage, SpanStatus, SpanType
from ..pricing import build_llm_usage
from ..tracer import _span_id, get_tracer

_original_generate_content = None
_original_generate_content_async = None
_patched = False


def _extract_model_name(instance: Any) -> str:
    return str(
        getattr(instance, "model_name", None)
        or getattr(instance, "_model_name", None)
        or "gemini-unknown"
    )


def _extract_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return str(text)
    candidates = getattr(response, "candidates", None)
    if not candidates:
        return ""
    chunks = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) if content is not None else None
        if not parts:
            continue
        for part in parts:
            value = getattr(part, "text", None)
            if value:
                chunks.append(str(value))
    return "\n".join(chunks).strip()


def _extract_usage(response: Any) -> tuple[int, int, int]:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return 0, 0, 0

    prompt_tokens = int(
        getattr(usage, "prompt_token_count", 0)
        or getattr(usage, "input_token_count", 0)
        or 0
    )
    completion_tokens = int(
        getattr(usage, "candidates_token_count", 0)
        or getattr(usage, "output_token_count", 0)
        or 0
    )
    total_tokens = int(
        getattr(usage, "total_token_count", 0)
        or (prompt_tokens + completion_tokens)
    )
    return prompt_tokens, completion_tokens, total_tokens


def _assign_usage(span: Any, model: str, prompt_tokens: int, completion_tokens: int, total_tokens: int) -> None:
    usage = build_llm_usage(
        model=model,
        provider="google",
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


def _is_streaming(kwargs: dict) -> bool:
    return bool(kwargs.get("stream"))


def _contents_arg(args: tuple, kwargs: dict) -> Any:
    if "contents" in kwargs:
        return kwargs["contents"]
    if len(args) >= 2:
        return args[1]
    if len(args) == 1:
        return args[0]
    return None


def _record_multimodal_inputs(tracer: Any, span: Any, contents: Any) -> tuple[int, int]:
    image_count = 0
    audio_count = 0
    for artifact in extract_gemini_multimodal_artifacts(contents):
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


def _iter_google_stream_sync(tracer, span, stream, model, image_count, audio_count):
    """Consume a sync Gemini streaming response and finish the span."""
    text_parts: list = []
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    try:
        for chunk in stream:
            # Accumulate text from each chunk.
            chunk_text = getattr(chunk, "text", None)
            if chunk_text:
                text_parts.append(str(chunk_text))
            # Usage metadata is typically present on the final chunk.
            usage = getattr(chunk, "usage_metadata", None)
            if usage:
                prompt_tokens = int(
                    getattr(usage, "prompt_token_count", 0)
                    or getattr(usage, "input_token_count", 0)
                    or 0
                )
                completion_tokens = int(
                    getattr(usage, "candidates_token_count", 0)
                    or getattr(usage, "output_token_count", 0)
                    or 0
                )
                total_tokens = int(getattr(usage, "total_token_count", 0) or (prompt_tokens + completion_tokens))
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
                metadata={"source": "google.generativeai.generate_content"},
            )
        _assign_usage(span, model, prompt_tokens, completion_tokens, total_tokens or prompt_tokens + completion_tokens)
        tracer.finish_span(span, SpanStatus.SUCCESS)


async def _iter_google_stream_async(tracer, span, stream, model, image_count, audio_count):
    """Consume an async Gemini streaming response and finish the span."""
    text_parts: list = []
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    try:
        async for chunk in stream:
            chunk_text = getattr(chunk, "text", None)
            if chunk_text:
                text_parts.append(str(chunk_text))
            usage = getattr(chunk, "usage_metadata", None)
            if usage:
                prompt_tokens = int(
                    getattr(usage, "prompt_token_count", 0)
                    or getattr(usage, "input_token_count", 0)
                    or 0
                )
                completion_tokens = int(
                    getattr(usage, "candidates_token_count", 0)
                    or getattr(usage, "output_token_count", 0)
                    or 0
                )
                total_tokens = int(getattr(usage, "total_token_count", 0) or (prompt_tokens + completion_tokens))
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
                metadata={"source": "google.generativeai.generate_content"},
            )
        _assign_usage(span, model, prompt_tokens, completion_tokens, total_tokens or prompt_tokens + completion_tokens)
        tracer.finish_span(span, SpanStatus.SUCCESS)


def _wrap_generate_content(original_func):
    @functools.wraps(original_func)
    def wrapper(*args, **kwargs):
        tracer = get_tracer()
        instance = args[0] if args else None
        model = _extract_model_name(instance)
        contents = _contents_arg(args, kwargs)
        is_streaming = _is_streaming(kwargs)

        span = tracer.create_span(
            name="google.generativeai.generate_content",
            span_type=SpanType.LLM,
            model=model,
            provider="google",
        )
        token = _span_id.set(span.span_id)
        input_payload = {"model": model, "contents": contents}
        tracer.record_artifact(
            span,
            role="input.message",
            content_type="application/json",
            content=input_payload,
            preview={"model": model},
        )
        image_count, audio_count = _record_multimodal_inputs(tracer, span, contents)
        _span_id.reset(token)

        try:
            response = original_func(*args, **kwargs)
            if is_streaming:
                return _iter_google_stream_sync(tracer, span, response, model, image_count, audio_count)
            prompt_tokens, completion_tokens, total_tokens = _extract_usage(response)
            _assign_usage(span, model, prompt_tokens, completion_tokens, total_tokens)
            output_text = _extract_text(response)
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
                    metadata={"source": "google.generativeai.generate_content"},
                )
            tracer.finish_span(span, SpanStatus.SUCCESS)
            return response
        except Exception as exc:
            tracer.finish_span(span, SpanStatus.ERROR, error=exc)
            raise

    return wrapper


def _wrap_generate_content_async(original_func):
    @functools.wraps(original_func)
    async def wrapper(*args, **kwargs):
        tracer = get_tracer()
        instance = args[0] if args else None
        model = _extract_model_name(instance)
        contents = _contents_arg(args, kwargs)
        is_streaming = _is_streaming(kwargs)

        span = tracer.create_span(
            name="google.generativeai.generate_content",
            span_type=SpanType.LLM,
            model=model,
            provider="google",
        )
        token = _span_id.set(span.span_id)
        input_payload = {"model": model, "contents": contents}
        tracer.record_artifact(
            span,
            role="input.message",
            content_type="application/json",
            content=input_payload,
            preview={"model": model},
        )
        image_count, audio_count = _record_multimodal_inputs(tracer, span, contents)
        _span_id.reset(token)

        try:
            response = await original_func(*args, **kwargs)
            if is_streaming:
                return _iter_google_stream_async(tracer, span, response, model, image_count, audio_count)
            prompt_tokens, completion_tokens, total_tokens = _extract_usage(response)
            _assign_usage(span, model, prompt_tokens, completion_tokens, total_tokens)
            output_text = _extract_text(response)
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
                    metadata={"source": "google.generativeai.generate_content"},
                )
            tracer.finish_span(span, SpanStatus.SUCCESS)
            return response
        except Exception as exc:
            tracer.finish_span(span, SpanStatus.ERROR, error=exc)
            raise

    return wrapper


def patch_google():
    """Patch google-generativeai model methods."""
    global _original_generate_content, _original_generate_content_async, _patched

    if _patched:
        return

    try:
        from google.generativeai import generative_models

        _original_generate_content = generative_models.GenerativeModel.generate_content
        generative_models.GenerativeModel.generate_content = _wrap_generate_content(_original_generate_content)

        maybe_async = getattr(generative_models.GenerativeModel, "generate_content_async", None)
        if maybe_async is not None:
            _original_generate_content_async = maybe_async
            generative_models.GenerativeModel.generate_content_async = _wrap_generate_content_async(maybe_async)

        _patched = True
        print("[OAT] Google Gemini patched successfully")
    except ImportError:
        print("[OAT] google-generativeai not installed, skipping patch")
    except Exception as exc:
        print(f"[OAT] Failed to patch Google Gemini: {exc}")


def unpatch_google():
    global _patched

    if not _patched:
        return

    try:
        from google.generativeai import generative_models

        if _original_generate_content:
            generative_models.GenerativeModel.generate_content = _original_generate_content
        if _original_generate_content_async:
            generative_models.GenerativeModel.generate_content_async = _original_generate_content_async
        _patched = False
    except Exception:
        pass
