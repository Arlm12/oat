"""
Base integration class for auto-instrumentation.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Type

# Registry of all integrations
_integrations: Dict[str, "Integration"] = {}


class Integration(ABC):
    """Base class for library integrations."""
    
    name: str = ""
    
    @abstractmethod
    def patch(self):
        """Apply monkey patches to instrument the library."""
        pass
    
    @abstractmethod
    def unpatch(self):
        """Remove monkey patches."""
        pass
    
    @classmethod
    def register(cls, integration_cls: Type["Integration"]):
        """Register an integration."""
        instance = integration_cls()
        _integrations[instance.name] = instance
        return integration_cls


def patch_all():
    """Patch all registered integrations."""
    # Built-in integrations.
    try:
        from .openai_integration import patch_openai

        patch_openai()
    except Exception:
        pass
    try:
        from .anthropic_integration import patch_anthropic

        patch_anthropic()
    except Exception:
        pass
    try:
        from .google_integration import patch_google

        patch_google()
    except Exception:
        pass
    try:
        from .ollama_integration import patch_ollama

        patch_ollama()
    except Exception:
        pass

    # Optional extensions registered through Integration subclasses.
    for name, integration in _integrations.items():
        try:
            integration.patch()
            print(f"[OAT] Patched {name}")
        except ImportError:
            pass  # Library not installed
        except Exception as e:
            print(f"[OAT] Failed to patch {name}: {e}")


def unpatch_all():
    """Unpatch all registered integrations."""
    try:
        from .openai_integration import unpatch_openai

        unpatch_openai()
    except Exception:
        pass
    try:
        from .anthropic_integration import unpatch_anthropic

        unpatch_anthropic()
    except Exception:
        pass
    try:
        from .google_integration import unpatch_google

        unpatch_google()
    except Exception:
        pass
    try:
        from .ollama_integration import unpatch_ollama

        unpatch_ollama()
    except Exception:
        pass

    for name, integration in _integrations.items():
        try:
            integration.unpatch()
        except Exception:
            pass
