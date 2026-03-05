"""Model pricing helpers for token/cost normalization."""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional, Tuple
import json
import os


# USD per 1K tokens. Keep this table intentionally compact and use pattern
# matching for families to avoid brittle one-model-per-release maintenance.
_BASE_PRICING: Dict[str, Dict[str, float]] = {
    # OpenAI
    "gpt-4o": {"prompt": 0.0025, "completion": 0.01},
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
    "gpt-4.1": {"prompt": 0.0020, "completion": 0.0080},
    "gpt-4.1-mini": {"prompt": 0.0004, "completion": 0.0016},
    "gpt-4.1-nano": {"prompt": 0.0001, "completion": 0.0004},
    "gpt-5": {"prompt": 0.0050, "completion": 0.0150},
    "gpt-5-mini": {"prompt": 0.0006, "completion": 0.0024},
    "gpt-5-nano": {"prompt": 0.0002, "completion": 0.0008},
    "o3": {"prompt": 0.0020, "completion": 0.0080},
    "o1": {"prompt": 0.0150, "completion": 0.0600},
    # Anthropic
    "claude-3-5-sonnet": {"prompt": 0.0030, "completion": 0.0150},
    "claude-3-7-sonnet": {"prompt": 0.0030, "completion": 0.0150},
    "claude-3-sonnet": {"prompt": 0.0030, "completion": 0.0150},
    "claude-3-haiku": {"prompt": 0.00025, "completion": 0.00125},
    "claude-3-opus": {"prompt": 0.0150, "completion": 0.0750},
    "claude-sonnet-4-5": {"prompt": 0.0030, "completion": 0.0150},
    "claude-sonnet-4": {"prompt": 0.0030, "completion": 0.0150},
    "claude-opus-4-1": {"prompt": 0.0150, "completion": 0.0750},
    # Google Gemini
    "gemini-2.5-pro": {"prompt": 0.0035, "completion": 0.0105},
    "gemini-2.5-flash": {"prompt": 0.00035, "completion": 0.00105},
    "gemini-2.5-flash-lite": {"prompt": 0.00010, "completion": 0.00030},
    "gemini-2.0-flash": {"prompt": 0.00035, "completion": 0.00105},
    # Mistral
    "mistral-large-latest": {"prompt": 0.0020, "completion": 0.0060},
    "mistral-medium-latest": {"prompt": 0.0007, "completion": 0.0021},
    "mistral-small-latest": {"prompt": 0.0002, "completion": 0.0006},
    "codestral-latest": {"prompt": 0.0003, "completion": 0.0009},
    "pixtral-large-latest": {"prompt": 0.0020, "completion": 0.0060},
    # DeepSeek / Qwen / Kimi / Moonshot (defaults can be overridden in file)
    "deepseek-chat": {"prompt": 0.00014, "completion": 0.00028},
    "deepseek-reasoner": {"prompt": 0.00055, "completion": 0.00219},
    "qwen-max": {"prompt": 0.0020, "completion": 0.0060},
    "qwen-plus": {"prompt": 0.0008, "completion": 0.0024},
    "qwen-turbo": {"prompt": 0.0003, "completion": 0.0009},
    "kimi-latest": {"prompt": 0.0012, "completion": 0.0036},
    "moonshot-v1-8k": {"prompt": 0.0012, "completion": 0.0036},
    "moonshot-v1-32k": {"prompt": 0.0012, "completion": 0.0036},
    "moonshot-v1-128k": {"prompt": 0.0012, "completion": 0.0036},
}


_PREFIX_FALLBACKS: Tuple[Tuple[str, Dict[str, float]], ...] = (
    ("gpt-5", {"prompt": 0.0050, "completion": 0.0150}),
    ("gpt-4o-mini", {"prompt": 0.00015, "completion": 0.0006}),
    ("gpt-4o", {"prompt": 0.0025, "completion": 0.01}),
    ("gpt-4.1-mini", {"prompt": 0.0004, "completion": 0.0016}),
    ("gpt-4.1", {"prompt": 0.0020, "completion": 0.0080}),
    ("o3", {"prompt": 0.0020, "completion": 0.0080}),
    ("o1", {"prompt": 0.0150, "completion": 0.0600}),
    ("claude-sonnet", {"prompt": 0.0030, "completion": 0.0150}),
    ("claude-haiku", {"prompt": 0.00025, "completion": 0.00125}),
    ("claude-opus", {"prompt": 0.0150, "completion": 0.0750}),
    ("gemini-2.5-pro", {"prompt": 0.0035, "completion": 0.0105}),
    ("gemini-2.5-flash-lite", {"prompt": 0.00010, "completion": 0.00030}),
    ("gemini-2.5-flash", {"prompt": 0.00035, "completion": 0.00105}),
    ("gemini-2.0-flash", {"prompt": 0.00035, "completion": 0.00105}),
    ("mistral-large", {"prompt": 0.0020, "completion": 0.0060}),
    ("mistral-medium", {"prompt": 0.0007, "completion": 0.0021}),
    ("mistral-small", {"prompt": 0.0002, "completion": 0.0006}),
    ("deepseek-chat", {"prompt": 0.00014, "completion": 0.00028}),
    ("deepseek-reasoner", {"prompt": 0.00055, "completion": 0.00219}),
    ("qwen-max", {"prompt": 0.0020, "completion": 0.0060}),
    ("qwen-plus", {"prompt": 0.0008, "completion": 0.0024}),
    ("qwen-turbo", {"prompt": 0.0003, "completion": 0.0009}),
    ("kimi", {"prompt": 0.0012, "completion": 0.0036}),
    ("moonshot", {"prompt": 0.0012, "completion": 0.0036}),
)


_ALIAS_MAP: Dict[str, str] = {
    "claude-sonnet-4-5-20250929": "claude-sonnet-4-5",
    "claude-sonnet-4-20250514": "claude-sonnet-4",
    "claude-opus-4-1-20250805": "claude-opus-4-1",
    "mistral-large": "mistral-large-latest",
    "mistral-medium": "mistral-medium-latest",
    "mistral-small": "mistral-small-latest",
}


_lock = Lock()
_overrides_cache: Optional[Dict[str, Dict[str, float]]] = None
_overrides_file: Optional[Path] = None
_overrides_mtime: Optional[float] = None


def _parse_price_pair(raw: Any) -> Optional[Dict[str, float]]:
    if not isinstance(raw, dict):
        return None
    prompt = raw.get("prompt")
    completion = raw.get("completion")
    if prompt is None:
        prompt = raw.get("prompt_tokens_per_1k")
    if completion is None:
        completion = raw.get("completion_tokens_per_1k")
    try:
        if prompt is None or completion is None:
            return None
        return {"prompt": float(prompt), "completion": float(completion)}
    except (TypeError, ValueError):
        return None


def _load_override_file(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        return {}
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        try:
            import yaml  # type: ignore
        except Exception:
            return {}
        data = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        return {}
    pricing = data.get("pricing", data)
    if not isinstance(pricing, dict):
        return {}

    out: Dict[str, Dict[str, float]] = {}
    for model, spec in pricing.items():
        pair = _parse_price_pair(spec)
        if pair:
            out[str(model).strip().lower()] = pair
    return out


def _default_override_path() -> Optional[Path]:
    env = os.getenv("OAT_PRICING_FILE")
    if env:
        return Path(env)
    for candidate in (Path("pricing_overrides.yaml"), Path("pricing_overrides.yml"), Path("pricing_overrides.json")):
        if candidate.exists():
            return candidate
    return None


def _get_overrides() -> Dict[str, Dict[str, float]]:
    global _overrides_cache, _overrides_file, _overrides_mtime
    with _lock:
        path = _default_override_path()
        if path is None:
            _overrides_cache = {}
            _overrides_file = None
            _overrides_mtime = None
            return {}
        mtime = path.stat().st_mtime if path.exists() else None
        if _overrides_cache is not None and _overrides_file == path and _overrides_mtime == mtime:
            return _overrides_cache
        _overrides_cache = _load_override_file(path)
        _overrides_file = path
        _overrides_mtime = mtime
        return _overrides_cache


def _normalize_model_lookup_key(model: Optional[str]) -> str:
    text = (model or "").strip().lower()
    if not text:
        return ""
    if ":" in text:
        left, right = text.split(":", 1)
        if left and right and left in {
            "openai",
            "anthropic",
            "google",
            "gemini",
            "ollama",
            "mistral",
            "moonshot",
            "kimi",
            "qwen",
            "openrouter",
            "groq",
            "deepseek",
            "self_hosted",
            "vllm",
        }:
            text = right
    if "/" in text:
        parts = [p for p in text.split("/") if p]
        if len(parts) >= 2:
            text = parts[-1]
    return _ALIAS_MAP.get(text, text)


def get_model_pricing_info(model: Optional[str], provider: Optional[str] = None) -> Dict[str, Any]:
    normalized = _normalize_model_lookup_key(model)
    table = dict(_BASE_PRICING)
    table.update(_get_overrides())

    if normalized in table:
        row = table[normalized]
        return {
            "model": model,
            "normalized_model": normalized,
            "provider": provider,
            "prompt_per_1k": float(row["prompt"]),
            "completion_per_1k": float(row["completion"]),
            "currency": "USD",
            "pricing_status": "known",
        }

    # Secondary heuristic: try prefix fallbacks.
    for prefix, costs in _PREFIX_FALLBACKS:
        if normalized.startswith(prefix):
            return {
                "model": model,
                "normalized_model": normalized,
                "provider": provider,
                "prompt_per_1k": float(costs["prompt"]),
                "completion_per_1k": float(costs["completion"]),
                "currency": "USD",
                "pricing_status": "known",
            }

    return {
        "model": model,
        "normalized_model": normalized,
        "provider": provider,
        "prompt_per_1k": 0.0,
        "completion_per_1k": 0.0,
        "currency": "USD",
        "pricing_status": "unknown",
    }


def calculate_cost(
    model: Optional[str],
    prompt_tokens: int,
    completion_tokens: int,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    info = get_model_pricing_info(model=model, provider=provider)
    prompt = int(prompt_tokens or 0)
    completion = int(completion_tokens or 0)
    total = prompt + completion
    prompt_cost = (prompt / 1000.0) * float(info["prompt_per_1k"])
    completion_cost = (completion / 1000.0) * float(info["completion_per_1k"])
    total_cost = prompt_cost + completion_cost
    return {
        "model": model,
        "provider": provider,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "prompt_cost": prompt_cost,
        "completion_cost": completion_cost,
        "total_cost": total_cost,
        "pricing_status": info["pricing_status"],
        "pricing": info,
    }


def build_llm_usage(
    model: Optional[str],
    prompt_tokens: int,
    completion_tokens: int,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    result = calculate_cost(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        provider=provider,
    )
    return {
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
        "total_tokens": result["total_tokens"],
        "prompt_cost": result["prompt_cost"],
        "completion_cost": result["completion_cost"],
        "total_cost": result["total_cost"],
        "pricing_status": result["pricing_status"],
    }
