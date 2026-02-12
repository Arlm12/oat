"""
Auto-instrumentation integrations for popular AI libraries.
"""

from .openai_integration import patch_openai, unpatch_openai
from .anthropic_integration import patch_anthropic, unpatch_anthropic
from .base import Integration, patch_all, unpatch_all

__all__ = [
    "Integration",
    "patch_openai",
    "unpatch_openai",
    "patch_anthropic",
    "unpatch_anthropic",
    "patch_all",
    "unpatch_all",
]

