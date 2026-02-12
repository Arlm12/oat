"""
OpenAI auto-instrumentation for OpenAgentTrace.

Automatically traces:
- chat.completions.create()
- embeddings.create()
- All async variants

Usage:
    from oat.integrations import patch_openai
    patch_openai()
    
    # Now all OpenAI calls are automatically traced
    response = openai.chat.completions.create(...)
"""

import functools
from typing import Any, Optional

from ..tracer import get_tracer, _span_id
from ..models import Span, SpanType, SpanStatus, LLMUsage
from ..pricing import calculate_cost as _calculate_cost_full

_original_chat_create = None
_original_chat_create_async = None
_original_embeddings_create = None
_original_embeddings_create_async = None
_patched = False


def _calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> tuple:
    """Calculate cost using centralized pricing module."""
    prompt_cost, completion_cost, _ = _calculate_cost_full(model, prompt_tokens, completion_tokens)
    return prompt_cost, completion_cost


def _extract_media_from_messages(messages: list) -> list:
    """Extract media metadata from OpenAI vision messages."""
    media_list = []
    
    try:
        from ..media import analyze_image, MediaMetadata
        
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "image_url":
                        image_url = part.get("image_url", {})
                        url = image_url.get("url", "")
                        detail = image_url.get("detail", "auto")
                        
                        # Analyze the image
                        metadata = analyze_image(url)
                        if metadata:
                            # Update token estimate based on detail level
                            if detail == "low":
                                metadata.estimated_tokens = 85
                            media_list.append(metadata.to_dict())
    except Exception:
        pass
    
    return media_list



def _wrap_chat_completion(original_func):
    """Wrap synchronous chat completion."""
    @functools.wraps(original_func)
    def wrapper(*args, **kwargs):
        tracer = get_tracer()
        
        # Extract model from kwargs or args
        model = kwargs.get('model', 'unknown')
        messages = kwargs.get('messages', [])
        
        span = tracer.create_span(
            name="openai.chat.completions.create",
            span_type=SpanType.LLM,
            model=model,
            service_provider="openai"
        )
        token = _span_id.set(span.span_id)
        
        # Capture input
        inputs = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get('temperature'),
            "max_tokens": kwargs.get('max_tokens'),
        }
        
        # Extract media from vision messages
        span.media_inputs = _extract_media_from_messages(messages)
        
        try:
            response = original_func(*args, **kwargs)
            
            # Extract usage
            if hasattr(response, 'usage') and response.usage:
                prompt_tokens = response.usage.prompt_tokens or 0
                completion_tokens = response.usage.completion_tokens or 0
                prompt_cost, completion_cost = _calculate_cost(model, prompt_tokens, completion_tokens)
                
                span.usage = LLMUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=response.usage.total_tokens or 0,
                    prompt_cost=prompt_cost,
                    completion_cost=completion_cost,
                    total_cost=prompt_cost + completion_cost,
                )
            
            span.finish(SpanStatus.SUCCESS)
            
            # Capture output
            outputs = None
            if hasattr(response, 'choices') and response.choices:
                outputs = {
                    "content": response.choices[0].message.content if response.choices[0].message else None,
                    "finish_reason": response.choices[0].finish_reason,
                }
            
            tracer._enqueue_span(span, inputs, outputs)
            return response
            
        except Exception as e:
            span.finish(SpanStatus.ERROR, error=e)
            tracer._enqueue_span(span, inputs, {"error": str(e)})
            raise
        finally:
            _span_id.reset(token)
    
    return wrapper


def _wrap_chat_completion_async(original_func):
    """Wrap async chat completion."""
    @functools.wraps(original_func)
    async def wrapper(*args, **kwargs):
        tracer = get_tracer()
        
        model = kwargs.get('model', 'unknown')
        messages = kwargs.get('messages', [])
        
        span = tracer.create_span(
            name="openai.chat.completions.create",
            span_type=SpanType.LLM,
            model=model,
            service_provider="openai"
        )
        token = _span_id.set(span.span_id)
        
        inputs = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get('temperature'),
            "max_tokens": kwargs.get('max_tokens'),
        }
        
        # Extract media from vision messages
        span.media_inputs = _extract_media_from_messages(messages)
        
        try:
            response = await original_func(*args, **kwargs)
            
            if hasattr(response, 'usage') and response.usage:
                prompt_tokens = response.usage.prompt_tokens or 0
                completion_tokens = response.usage.completion_tokens or 0
                prompt_cost, completion_cost = _calculate_cost(model, prompt_tokens, completion_tokens)
                
                span.usage = LLMUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=response.usage.total_tokens or 0,
                    prompt_cost=prompt_cost,
                    completion_cost=completion_cost,
                    total_cost=prompt_cost + completion_cost,
                )
            
            span.finish(SpanStatus.SUCCESS)
            
            outputs = None
            if hasattr(response, 'choices') and response.choices:
                outputs = {
                    "content": response.choices[0].message.content if response.choices[0].message else None,
                    "finish_reason": response.choices[0].finish_reason,
                }
            
            tracer._enqueue_span(span, inputs, outputs)
            return response
            
        except Exception as e:
            span.finish(SpanStatus.ERROR, error=e)
            tracer._enqueue_span(span, inputs, {"error": str(e)})
            raise
        finally:
            _span_id.reset(token)
    
    return wrapper


def _wrap_embeddings(original_func):
    """Wrap synchronous embeddings."""
    @functools.wraps(original_func)
    def wrapper(*args, **kwargs):
        tracer = get_tracer()
        
        model = kwargs.get('model', 'unknown')
        input_text = kwargs.get('input', '')
        
        span = tracer.create_span(
            name="openai.embeddings.create",
            span_type=SpanType.EMBEDDING,
            model=model,
            service_provider="openai"
        )
        token = _span_id.set(span.span_id)
        
        inputs = {"model": model, "input": input_text[:500] if isinstance(input_text, str) else f"[{len(input_text)} items]"}
        
        try:
            response = original_func(*args, **kwargs)
            
            if hasattr(response, 'usage') and response.usage:
                prompt_tokens = response.usage.prompt_tokens or 0
                prompt_cost, _ = _calculate_cost(model, prompt_tokens, 0)
                
                span.usage = LLMUsage(
                    prompt_tokens=prompt_tokens,
                    total_tokens=prompt_tokens,
                    prompt_cost=prompt_cost,
                    total_cost=prompt_cost,
                )
            
            span.finish(SpanStatus.SUCCESS)
            outputs = {"dimensions": len(response.data[0].embedding) if response.data else 0}
            tracer._enqueue_span(span, inputs, outputs)
            return response
            
        except Exception as e:
            span.finish(SpanStatus.ERROR, error=e)
            tracer._enqueue_span(span, inputs, {"error": str(e)})
            raise
        finally:
            _span_id.reset(token)
    
    return wrapper


def _wrap_embeddings_async(original_func):
    """Wrap async embeddings."""
    @functools.wraps(original_func)
    async def wrapper(*args, **kwargs):
        tracer = get_tracer()
        
        model = kwargs.get('model', 'unknown')
        input_text = kwargs.get('input', '')
        
        span = tracer.create_span(
            name="openai.embeddings.create",
            span_type=SpanType.EMBEDDING,
            model=model
        )
        token = _span_id.set(span.span_id)
        
        inputs = {"model": model, "input": input_text[:500] if isinstance(input_text, str) else f"[{len(input_text)} items]"}
        
        try:
            response = await original_func(*args, **kwargs)
            
            if hasattr(response, 'usage') and response.usage:
                prompt_tokens = response.usage.prompt_tokens or 0
                prompt_cost, _ = _calculate_cost(model, prompt_tokens, 0)
                
                span.usage = LLMUsage(
                    prompt_tokens=prompt_tokens,
                    total_tokens=prompt_tokens,
                    prompt_cost=prompt_cost,
                    total_cost=prompt_cost,
                )
            
            span.finish(SpanStatus.SUCCESS)
            outputs = {"dimensions": len(response.data[0].embedding) if response.data else 0}
            tracer._enqueue_span(span, inputs, outputs)
            return response
            
        except Exception as e:
            span.finish(SpanStatus.ERROR, error=e)
            tracer._enqueue_span(span, inputs, {"error": str(e)})
            raise
        finally:
            _span_id.reset(token)
    
    return wrapper


def patch_openai():
    """
    Patch the OpenAI library to automatically trace all API calls.
    
    Call this once at application startup before making any OpenAI calls.
    """
    global _original_chat_create, _original_chat_create_async
    global _original_embeddings_create, _original_embeddings_create_async
    global _patched
    
    if _patched:
        return
    
    try:
        import openai
        from openai.resources.chat import completions as chat_completions
        from openai.resources import embeddings
        
        # Patch sync chat completions
        _original_chat_create = chat_completions.Completions.create
        chat_completions.Completions.create = _wrap_chat_completion(_original_chat_create)
        
        # Patch async chat completions
        _original_chat_create_async = chat_completions.AsyncCompletions.create
        chat_completions.AsyncCompletions.create = _wrap_chat_completion_async(_original_chat_create_async)
        
        # Patch sync embeddings
        _original_embeddings_create = embeddings.Embeddings.create
        embeddings.Embeddings.create = _wrap_embeddings(_original_embeddings_create)
        
        # Patch async embeddings
        _original_embeddings_create_async = embeddings.AsyncEmbeddings.create
        embeddings.AsyncEmbeddings.create = _wrap_embeddings_async(_original_embeddings_create_async)
        
        _patched = True
        print("[OAT] OpenAI patched successfully")
        
    except ImportError:
        print("[OAT] OpenAI not installed, skipping patch")
    except Exception as e:
        print(f"[OAT] Failed to patch OpenAI: {e}")


def unpatch_openai():
    """Remove OpenAI patches."""
    global _patched
    
    if not _patched:
        return
    
    try:
        from openai.resources.chat import completions as chat_completions
        from openai.resources import embeddings
        
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
