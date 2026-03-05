"""Provider normalization and detection helpers for OAT."""

from __future__ import annotations

from typing import Optional


_KNOWN_PROVIDERS = {
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
    "openai_compatible",
}


def normalize_provider_name(provider: Optional[str]) -> Optional[str]:
    if not provider:
        return None
    value = provider.strip().lower().replace("-", "_")
    aliases = {
        "genai": "google",
        "gemini": "google",
        "moonshotai": "moonshot",
        "open_ai": "openai",
        "openrouter_ai": "openrouter",
        "selfhosted": "self_hosted",
        "self_host": "self_hosted",
    }
    return aliases.get(value, value)


def resolve_service_provider(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    provider_hint: Optional[str] = None,
    default: str = "openai",
) -> str:
    hint = normalize_provider_name(provider_hint)
    if hint in _KNOWN_PROVIDERS:
        return hint

    base = (base_url or "").strip().lower()
    if base:
        if "openrouter.ai" in base:
            return "openrouter"
        if "api.groq.com" in base:
            return "groq"
        if "api.deepseek.com" in base:
            return "deepseek"
        if "dashscope.aliyuncs.com" in base:
            return "qwen"
        if "moonshot.cn" in base:
            return "moonshot"
        if "api.mistral.ai" in base:
            return "mistral"
        if "localhost:11434" in base or "127.0.0.1:11434" in base or ":11434" in base:
            return "ollama"
        if "/v1beta" in base and "generativelanguage.googleapis.com" in base:
            return "google"
        if "vllm" in base:
            return "vllm"

    raw_model = (model or "").strip().lower()
    if raw_model:
        if raw_model.startswith("openrouter/") or raw_model.startswith("openrouter:"):
            return "openrouter"
        if raw_model.startswith("openai/") or raw_model.startswith("openai:"):
            return "openai"
        if raw_model.startswith("anthropic/") or raw_model.startswith("anthropic:"):
            return "anthropic"
        if raw_model.startswith("google/") or raw_model.startswith("gemini/") or raw_model.startswith("google:"):
            return "google"
        if raw_model.startswith("ollama/") or raw_model.startswith("ollama:"):
            return "ollama"
        if raw_model.startswith("groq/") or raw_model.startswith("groq:"):
            return "groq"
        if raw_model.startswith("deepseek/") or raw_model.startswith("deepseek:") or raw_model.startswith("deepseek-"):
            return "deepseek"
        if raw_model.startswith("qwen/") or raw_model.startswith("qwen:") or raw_model.startswith("qwen-"):
            return "qwen"
        if raw_model.startswith("kimi/") or raw_model.startswith("kimi:") or raw_model.startswith("kimi-"):
            return "kimi"
        if raw_model.startswith("moonshot/") or raw_model.startswith("moonshot:") or raw_model.startswith("moonshot-"):
            return "moonshot"
        if raw_model.startswith("mistral/") or raw_model.startswith("mistral:") or raw_model.startswith("mistral-"):
            return "mistral"
        if raw_model.startswith("vllm/") or raw_model.startswith("vllm:"):
            return "vllm"
        if raw_model.startswith("claude-"):
            return "anthropic"
        if raw_model.startswith("gemini-"):
            return "google"
        if raw_model.startswith("gpt-") or raw_model.startswith("o1-") or raw_model.startswith("o3-"):
            return "openai"
        if raw_model.startswith("llama") and base:
            if "groq" in base:
                return "groq"
            if "11434" in base:
                return "ollama"

    return normalize_provider_name(default) or "openai"


def normalize_model_name(model: Optional[str]) -> str:
    value = (model or "").strip()
    if not value:
        return ""
    lower = value.lower()

    # Common provider-id wrappers.
    if ":" in lower:
        left, right = lower.split(":", 1)
        if left in _KNOWN_PROVIDERS and right:
            lower = right
    if "/" in lower:
        parts = [p for p in lower.split("/") if p]
        if len(parts) >= 2:
            if parts[0] in _KNOWN_PROVIDERS:
                lower = "/".join(parts[1:])
            if "/" in lower:
                # openai/gpt-4o-mini -> gpt-4o-mini
                lower = lower.split("/")[-1]

    return lower
