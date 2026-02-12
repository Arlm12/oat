"""
OpenAgentTrace Server - FastAPI backend for the observability dashboard.
"""

from .main import app, create_app

__all__ = ["app", "create_app"]
