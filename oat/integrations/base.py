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
    for name, integration in _integrations.items():
        try:
            integration.unpatch()
        except Exception:
            pass
