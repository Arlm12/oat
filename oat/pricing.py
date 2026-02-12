"""
Pricing module for OpenAgentTrace.
Loads model pricing from tracer.yaml and calculates costs.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import yaml

# Default pricing (per 1K tokens) if tracer.yaml not found
DEFAULT_PRICING = {
    # OpenAI
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "o1-preview": {"input": 0.015, "output": 0.06},
    "o1-mini": {"input": 0.003, "output": 0.012},
    # Anthropic
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    # Google
    "gemini-pro": {"input": 0.00025, "output": 0.0005},
    "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
    "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
    # Mistral
    "mistral-large": {"input": 0.004, "output": 0.012},
    "mistral-medium": {"input": 0.0027, "output": 0.0081},
    "mistral-small": {"input": 0.001, "output": 0.003},
}

_pricing_cache: Optional[Dict] = None


def _load_pricing() -> Dict:
    """Load pricing from tracer.yaml or fall back to defaults."""
    global _pricing_cache
    
    if _pricing_cache is not None:
        return _pricing_cache
    
    # Try to find tracer.yaml
    search_paths = [
        Path.cwd() / "tracer.yaml",
        Path.cwd().parent / "tracer.yaml",
        Path(__file__).parent.parent / "tracer.yaml",
    ]
    
    for path in search_paths:
        if path.exists():
            try:
                with open(path) as f:
                    config = yaml.safe_load(f)
                    if config and "cost_config" in config and "models" in config["cost_config"]:
                        _pricing_cache = config["cost_config"]["models"]
                        return _pricing_cache
            except Exception:
                pass
    
    # Fall back to defaults
    _pricing_cache = DEFAULT_PRICING
    return _pricing_cache


def get_model_pricing(model: str) -> Tuple[float, float]:
    """
    Get input/output pricing per 1K tokens for a model.
    
    Returns:
        Tuple of (input_cost_per_1k, output_cost_per_1k)
    """
    pricing = _load_pricing()
    
    # Normalize model name (handle variations)
    model_lower = model.lower() if model else ""
    
    # Direct match
    if model_lower in pricing:
        p = pricing[model_lower]
        return (p.get("input", 0), p.get("output", 0))
    
    # Fuzzy match (e.g., "gpt-4o-2024-05-13" -> "gpt-4o")
    for key in pricing:
        if key in model_lower or model_lower.startswith(key):
            p = pricing[key]
            return (p.get("input", 0), p.get("output", 0))
    
    return (0.0, 0.0)


def calculate_cost(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    total_tokens: int = 0
) -> Tuple[float, float, float]:
    """
    Calculate cost for a model call.
    
    Args:
        model: Model name (e.g., "gpt-4o", "claude-3-sonnet")
        input_tokens: Number of input/prompt tokens
        output_tokens: Number of output/completion tokens
        total_tokens: Total tokens (used if input/output not provided)
    
    Returns:
        Tuple of (input_cost, output_cost, total_cost)
    """
    input_rate, output_rate = get_model_pricing(model)
    
    # If only total_tokens provided, estimate split (60/40)
    if total_tokens > 0 and input_tokens == 0 and output_tokens == 0:
        input_tokens = int(total_tokens * 0.6)
        output_tokens = total_tokens - input_tokens
    
    input_cost = (input_tokens / 1000) * input_rate
    output_cost = (output_tokens / 1000) * output_rate
    total_cost = input_cost + output_cost
    
    return (input_cost, output_cost, total_cost)


def get_available_models() -> Dict[str, Dict]:
    """Get all available models with their pricing."""
    return _load_pricing()
