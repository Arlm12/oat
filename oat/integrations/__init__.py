"""
Auto-instrumentation integrations for popular AI libraries.
"""

from .openai_integration import patch_openai, unpatch_openai
from .anthropic_integration import patch_anthropic, unpatch_anthropic
from .google_integration import patch_google, unpatch_google
from .ollama_integration import patch_ollama, unpatch_ollama
from .base import Integration, patch_all, unpatch_all

__all__ = [
    "Integration",
    "patch_openai",
    "unpatch_openai",
    "patch_anthropic",
    "unpatch_anthropic",
    "patch_google",
    "unpatch_google",
    "patch_ollama",
    "unpatch_ollama",
    "patch_all",
    "unpatch_all",
]
