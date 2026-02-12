"""
Anthropic Claude auto-instrumentation for OpenAgentTrace.

Automatically traces:
- messages.create()
- Async variants

Usage:
    from oat.integrations import patch_anthropic
    patch_anthropic()
    
    # Now all Anthropic calls are automatically traced
    response = client.messages.create(...)
"""

import functools
from typing import Any, Optional

from ..tracer import get_tracer, _span_id
from ..models import Span, SpanType, SpanStatus, LLMUsage

_original_messages_create = None
_original_messages_create_async = None
_patched = False


# Cost per 1K tokens for Claude models (approximate, as of 2024)
MODEL_COSTS = {
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
    "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    "claude-2.1": {"input": 0.008, "output": 0.024},
    "claude-2.0": {"input": 0.008, "output": 0.024},
    "claude-instant-1.2": {"input": 0.0008, "output": 0.0024},
}


def _calculate_cost(model: str, input_tokens: int, output_tokens: int) -> tuple:
    """Calculate cost based on model and token counts."""
    costs = MODEL_COSTS.get(model, {"input": 0, "output": 0})
    
    # Check for partial matches
    for model_key, model_costs in MODEL_COSTS.items():
        if model_key in model.lower():
            costs = model_costs
            break
    
    input_cost = (input_tokens / 1000) * costs["input"]
    output_cost = (output_tokens / 1000) * costs["output"]
    return input_cost, output_cost


def _extract_content_text(content) -> Optional[str]:
    """Extract text content from Anthropic response content blocks."""
    if not content:
        return None
    
    texts = []
    for block in content:
        if hasattr(block, 'text'):
            texts.append(block.text)
        elif hasattr(block, 'type') and block.type == 'text':
            texts.append(getattr(block, 'text', ''))
    
    return '\n'.join(texts) if texts else None


def _wrap_messages_create(original_func):
    """Wrap synchronous messages.create."""
    @functools.wraps(original_func)
    def wrapper(*args, **kwargs):
        tracer = get_tracer()
        
        # Extract model from kwargs
        model = kwargs.get('model', 'unknown')
        messages = kwargs.get('messages', [])
        system = kwargs.get('system', None)
        
        span = tracer.create_span(
            name="anthropic.messages.create",
            span_type=SpanType.LLM,
            model=model,
            service_provider="anthropic"
        )
        token = _span_id.set(span.span_id)
        
        # Capture input
        inputs = {
            "model": model,
            "messages": messages,
            "system": system[:500] if system and isinstance(system, str) else system,
            "max_tokens": kwargs.get('max_tokens'),
            "temperature": kwargs.get('temperature'),
        }
        
        try:
            response = original_func(*args, **kwargs)
            
            # Extract usage
            if hasattr(response, 'usage') and response.usage:
                input_tokens = getattr(response.usage, 'input_tokens', 0) or 0
                output_tokens = getattr(response.usage, 'output_tokens', 0) or 0
                input_cost, output_cost = _calculate_cost(model, input_tokens, output_tokens)
                
                span.usage = LLMUsage(
                    prompt_tokens=input_tokens,
                    completion_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                    prompt_cost=input_cost,
                    completion_cost=output_cost,
                    total_cost=input_cost + output_cost,
                )
            
            span.finish(SpanStatus.SUCCESS)
            
            # Capture output
            outputs = None
            if hasattr(response, 'content'):
                outputs = {
                    "content": _extract_content_text(response.content),
                    "stop_reason": getattr(response, 'stop_reason', None),
                    "model": getattr(response, 'model', model),
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


def _wrap_messages_create_async(original_func):
    """Wrap async messages.create."""
    @functools.wraps(original_func)
    async def wrapper(*args, **kwargs):
        tracer = get_tracer()
        
        model = kwargs.get('model', 'unknown')
        messages = kwargs.get('messages', [])
        system = kwargs.get('system', None)
        
        span = tracer.create_span(
            name="anthropic.messages.create",
            span_type=SpanType.LLM,
            model=model,
            service_provider="anthropic"
        )
        token = _span_id.set(span.span_id)
        
        inputs = {
            "model": model,
            "messages": messages,
            "system": system[:500] if system and isinstance(system, str) else system,
            "max_tokens": kwargs.get('max_tokens'),
            "temperature": kwargs.get('temperature'),
        }
        
        try:
            response = await original_func(*args, **kwargs)
            
            if hasattr(response, 'usage') and response.usage:
                input_tokens = getattr(response.usage, 'input_tokens', 0) or 0
                output_tokens = getattr(response.usage, 'output_tokens', 0) or 0
                input_cost, output_cost = _calculate_cost(model, input_tokens, output_tokens)
                
                span.usage = LLMUsage(
                    prompt_tokens=input_tokens,
                    completion_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                    prompt_cost=input_cost,
                    completion_cost=output_cost,
                    total_cost=input_cost + output_cost,
                )
            
            span.finish(SpanStatus.SUCCESS)
            
            outputs = None
            if hasattr(response, 'content'):
                outputs = {
                    "content": _extract_content_text(response.content),
                    "stop_reason": getattr(response, 'stop_reason', None),
                    "model": getattr(response, 'model', model),
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


def patch_anthropic():
    """
    Patch the Anthropic library to automatically trace all API calls.
    
    Call this once at application startup before making any Anthropic calls.
    """
    global _original_messages_create, _original_messages_create_async
    global _patched
    
    if _patched:
        return
    
    try:
        import anthropic
        from anthropic.resources import messages
        
        # Patch sync messages.create
        _original_messages_create = messages.Messages.create
        messages.Messages.create = _wrap_messages_create(_original_messages_create)
        
        # Patch async messages.create
        _original_messages_create_async = messages.AsyncMessages.create
        messages.AsyncMessages.create = _wrap_messages_create_async(_original_messages_create_async)
        
        _patched = True
        print("[OAT] Anthropic patched successfully")
        
    except ImportError:
        print("[OAT] Anthropic not installed, skipping patch")
    except Exception as e:
        print(f"[OAT] Failed to patch Anthropic: {e}")


def unpatch_anthropic():
    """Remove Anthropic patches."""
    global _patched
    
    if not _patched:
        return
    
    try:
        from anthropic.resources import messages
        
        if _original_messages_create:
            messages.Messages.create = _original_messages_create
        if _original_messages_create_async:
            messages.AsyncMessages.create = _original_messages_create_async
        
        _patched = False
        
    except Exception:
        pass
