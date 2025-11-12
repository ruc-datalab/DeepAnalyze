"""
DeepAnalyze API Package
OpenAI-compatible API server for DeepAnalyze model
"""

__version__ = "1.0.0"
__title__ = "DeepAnalyze OpenAI-Compatible API"

from .main import create_app, main

__all__ = ["create_app", "main"]